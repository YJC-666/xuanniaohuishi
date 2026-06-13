#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
import sys
from ultralytics import YOLO
from time import time
import tf.transformations
import math
import json

from std_msgs.msg import Header, String, Bool
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes

class Yolo_Dect:
    def __init__(self):
        # jiazai canshu
        self.weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov8/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = float(rospy.get_param('~conf', '0.5'))
        self.visualize = rospy.get_param('~visualize', 'True')

        # xiangji neican shezhi (genju yonghu tigong de canshu)
        self.camera_matrix = np.array([[369.502083, 0.0, 640.0],
                                      [0.0, 369.502083, 360.0],
                                      [0.0, 0.0, 1.0]])
        self.fx = 369.502083  # jiaoju x
        self.fy = 369.502083  # jiaoju y
        self.cx = 640.0       # zhudian x
        self.cy = 360.0       # zhudian y
        
        # shiji shiyong de fenbianlv 1280x720
        self.cx_actual = 640.0  # 1280/2
        self.cy_actual = 360.0  # 720/2
        
        # wurenjiGaodu he jiance quyu shezhi
        self.drone_height = 1.22  # wurenji juli dimian gaodu (mi)
        self.roi_size_real = 0.5  # jiance quyu bianzhang (mi): 50cm*50cm
        
        # jisuan jiance quyu zai xiangsu zuobiao xi zhong de fanwei
        self.calculate_roi_pixels()

        # shebei xuanze
        self.device = 'cpu' if rospy.get_param('/use_cpu', 'false') else 'cuda'

        # moxing jiazai
        try:
            self.model = YOLO(self.weight_path, task='detect')
            if self.weight_path.endswith('.pt'):  # 仅PyTorch模型需要特殊处理
                self.model.fuse()
                self.model.to(self.device)
            rospy.loginfo(f"chenggong jiazai moxing: {self.weight_path}")
        except Exception as e:
            rospy.logerr(f"moxing jiazai shibai: {str(e)}")
            sys.exit(1)

        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False
        self.classes_colors = {}

        # ROS tongxin shezhi
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                        queue_size=1, buff_size=52428800)
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)
        self.image_pub = rospy.Publisher('/yolov8/detection_image', Image, queue_size=1)
        self.xy_pub = rospy.Publisher('/yolov8/pub_image_xy', BoundingBox, queue_size=1)
        # 新的话题发布器 - 发布带区域ID的检测结果
        self.region_detection_pub = rospy.Publisher('/region_wildlife_detection', String, queue_size=1)
        # 全局目标里程计发布器
        self.global_target_odom_pub = rospy.Publisher('/yolov8/global_target_odom', Odometry, queue_size=1)
        
        # 新增话题发布器 - 用于发布动物id和值
        self.id_dongwu_value_pub = rospy.Publisher('/id_dongwu_value', String, queue_size=1)
        
        # 新增话题发布器 - 用于发送扫描信号
        self.sao_miao_pub = rospy.Publisher('/sao_miao', Bool, queue_size=1)
        
        # dingyue wurenji lichengji xinxi
        self.drone_odom_sub = rospy.Subscriber('iris_0/mavros/local_position/odom', Odometry, self.odom_callback, queue_size=1)
        
        # wurenji dangqian lichengji xinxi
        self.current_drone_odom = None
        self.drone_height_from_odom = 0.0
        
        # 区域判断变量
        self.region_tolerance = 0.25  # 35cm容差范围，提高到比0.25更大，让区域判断更宽松
        self.detected_regions = {}  # 用于记录检测到的区域
        
        # 订阅无人机位置信息用于区域判断
        self.drone_position = [0, 0, 0]  # [x, y, z]
        self.current_region_id = None  # 当前所在区域ID
        
        # 记录已经识别过的区域和动物类型，避免重复发送
        self.recognized_regions = set()  # 记录已经发布过的区域ID
        
        # 记录当前ROI中的动物及其框
        self.roi_animal_boxes = {}
        
        # 添加连续识别计数器
        self.consecutive_detections = {}  # 格式: {区域ID: {动物类型: 连续帧数}}
        self.required_consecutive_frames = 5  # 需要连续识别6帧以上才算有效
        
        # 添加区域进出跟踪变量
        self.region_animals = {}  # 格式: {区域ID: {动物类型: 数量}}
        self.validated_animals = {}  # 已经在区域中验证过的动物 {区域ID: {动物类型: True}}
        self.previous_region_id = None  # 记录上一个区域ID，用于检测区域变化
        self.region_exit_detection = True  # 启用区域退出检测 (仅在区域退出时发布数据)
        self.required_region_frames = 7  # 在区域内需要连续检测的帧数
        
        # 记录已经访问过的区域，防止重复检测
        self.visited_regions = set()  # 存储已经访问过的区域ID
        
        # 记录已发送扫描信号的区域，避免重复发送
        self.scan_sent_regions = set()  # 存储已发送扫描信号的区域ID
        
        # 创建订阅器 - 订阅无人机位置
        
        rospy.loginfo(f"YOLOv8 ROS jiedian yi chushihua, jiance quyu: {self.roi_x1}-{self.roi_x2}, {self.roi_y1}-{self.roi_y2}")
        rospy.loginfo("dengdai tuxiang shuru...")
        while not self.getImageStatus and not rospy.is_shutdown():
            rospy.sleep(0.1)
    
    def odom_callback(self, odom_msg):
        """处理无人机里程计信息"""
        try:
            self.current_drone_odom = odom_msg
            # 更新无人机位置
            self.drone_position = [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z
            ]
            
            # 从里程计中获取高度信息
            self.drone_height_from_odom = odom_msg.pose.pose.position.z
            
            # 更新高度设置（如果需要）
            if abs(self.drone_height_from_odom - self.drone_height) > 0.1:
                self.drone_height = max(0.5, self.drone_height_from_odom)  # 最小高度0.5m
                self.calculate_roi_pixels()  # 重新计算ROI
                
            # 判断当前所在区域
            previous_region = self.current_region_id
            self.current_region_id = self.determine_current_region()
            
            # 当区域发生变化时输出日志
            if previous_region != self.current_region_id:
                rospy.loginfo(f"无人机区域变化: {previous_region} -> {self.current_region_id}, 位置: ({self.drone_position[0]:.2f}, {self.drone_position[1]:.2f}, {self.drone_position[2]:.2f})")
                
                # 检查是否从有效区域退出
                if previous_region and previous_region in self.region_animals:
                    # 从区域退出，发送统计数据
                    self.process_region_exit(previous_region)
                    # 将退出的区域标记为已访问
                    self.visited_regions.add(previous_region)
                
                # 检查是否进入了已访问过的区域
                is_revisit = self.current_region_id in self.visited_regions
                if is_revisit and self.current_region_id:
                    rospy.logwarn(f"重新访问已处理过的区域: {self.current_region_id}，将忽略此区域内的新检测")
                
                # 重置当前区域的检测计数
                self.current_region_detection_count = 0
                
                # 验证坐标映射是否正确
                if self.current_region_id:
                    rospy.loginfo(f"坐标验证: x={self.drone_position[0]:.2f}, y={self.drone_position[1]:.2f} -> {self.current_region_id}")
                
        except Exception as e:
            rospy.logerr(f"处理里程计信息错误: {str(e)}")
            
    def process_region_exit(self, region_id):
        """处理从区域退出时的统计和发布"""
        try:
            if not region_id or region_id not in self.region_animals:
                return
                
            # 获取该区域内记录的动物
            region_animals = self.region_animals[region_id]
            
            # 检查是否有满足条件的动物（超过所需帧数）
            valid_animals = {}
            for animal_type, count in region_animals.items():
                if animal_type in self.validated_animals.get(region_id, {}) and self.validated_animals[region_id][animal_type]:
                    valid_animals[animal_type] = count
                    
            # 如果有有效动物，发送统计数据
            if valid_animals:
                # 构建JSON格式的消息
                detection_data = {
                    "region_id": region_id,
                    "detections": valid_animals
                    # 删除mode标记，只保留一种发布模式
                }
                
                # 转换为JSON字符串
                detection_msg = json.dumps(detection_data, ensure_ascii=False)
                
                # 发布检测结果到两个话题
                self.region_detection_pub.publish(String(detection_msg))
                self.id_dongwu_value_pub.publish(String(detection_msg))
                
                rospy.loginfo(f"【仅区域退出发布】发布区域 {region_id} 累积动物数据: {detection_msg}")
                
                # 详细输出每种动物的统计
                for animal_type, count in valid_animals.items():
                    rospy.loginfo(f"  - {animal_type}: {count}只 (已验证超过{self.required_region_frames}帧)")
                
            else:
                rospy.loginfo(f"区域退出: 区域 {region_id} 无有效动物数据 (未满足连续{self.required_region_frames}帧要求)")
                
            # 清除该区域的记录
            if region_id in self.region_animals:
                del self.region_animals[region_id]
            if region_id in self.validated_animals:
                del self.validated_animals[region_id]
                
        except Exception as e:
            rospy.logerr(f"处理区域退出错误: {str(e)}")
    
    def calculate_roi_pixels(self):
        """
        genju xiangji neican, wurenji gaodu he jiance quyu daxiao jisuan xiangsu zuobiao fanwei
        """
        # jisuan dimian shang 50cm duiying de xiangsu daxiao
        # shiyong xiangsi sanjiaoxing yuanli: xiangsu daxiao = (jiaoju * shiji daxiao) / juli
        pixel_size_x = (self.fx * self.roi_size_real) / self.drone_height
        pixel_size_y = (self.fy * self.roi_size_real) / self.drone_height
        
        # 使用裁剪后图像的中心作为ROI中心（640x480裁剪图像）
        image_width = 640   # 裁剪后的宽度
        image_height = 480  # 裁剪后的高度
        image_center_x = image_width / 2.0   # 裁剪图像的中心x
        image_center_y = image_height / 2.0  # 裁剪图像的中心y
        
        # jisuan jiance quyu de xiangsu zuobiao fanwei (严格以图像中心为基准)
        half_roi_x = int(pixel_size_x / 2)
        half_roi_y = int(pixel_size_y / 2)
        
        # 根据裁剪后的图像尺寸计算ROI边界
        self.roi_x1 = max(0, int(image_center_x - half_roi_x))
        self.roi_x2 = min(image_width, int(image_center_x + half_roi_x))
        self.roi_y1 = max(0, int(image_center_y - half_roi_y))
        self.roi_y2 = min(image_height, int(image_center_y + half_roi_y))
        
        rospy.loginfo(f"jisuan dedao de jiance quyu xiangsu fanwei: x[{self.roi_x1}-{self.roi_x2}], y[{self.roi_y1}-{self.roi_y2}]")
        rospy.loginfo(f"jiance quyu xiangsu daxiao: {self.roi_x2-self.roi_x1}x{self.roi_y2-self.roi_y1}")
        rospy.loginfo(f"caijian houde tuxiang daxiao: {image_width}x{image_height}")
        rospy.loginfo(f"tuxiang zhongxin dian: ({image_center_x}, {image_center_y})")

    def image_callback(self, image):
        try:
            self.boundingBoxes = BoundingBoxes()
            self.boundingBoxes.header = image.header
            self.boundingBoxes.image_header = image.header
            self.getImageStatus = True
            
            # ROS tuxiang zhuan OpenCV geshi
            self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width, -1)
            # baochi BGR geshi, yinwei OpenCV he ROS dou shiyong BGR geshi
            # self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

            # dui zhengge tuxiang jinxing tuili (zidong chuli ONNX/PyTorch geshi)
            results = self.model.predict(
                self.color_image,
                show=False,
                conf=0.3,
                device=self.device if self.weight_path.endswith('.pt') else None
            )
            
            self.dectshow(results, image.height, image.width)

        except Exception as e:
            rospy.logerr(f"tuili cuowu: {str(e)}")
            self.getImageStatus = False

    def dectshow(self, results, height, width):
        try:
            # chuangjian wanzheng tuxiang yongyu xianshi
            self.frame = self.color_image.copy()
            
            fps = 1000.0 / results[0].speed['inference']
            cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 绘制动态ROI区域框
            cv2.rectangle(self.frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 0, 255), 2)
            roi_text = f"ROI: {self.roi_x2-self.roi_x1}x{self.roi_y2-self.roi_y1}px, {self.roi_size_real*100:.0f}x{self.roi_size_real*100:.0f}cm"
            cv2.putText(self.frame, roi_text, (self.roi_x1, self.roi_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 绘制容差区域圆圈(0.35m半径)
            if self.current_region_id:
                # 计算到方格中心的距离
                drone_x, drone_y, _ = self.drone_position
                grid_col, grid_row = self.global_to_grid_coords(drone_x, drone_y)
                center_x = grid_col * 0.5
                center_y = grid_row * 0.5
                
                # 检查当前区域是否已访问过
                is_revisit = self.current_region_id in self.visited_regions
                region_color = (0, 0, 255) if is_revisit else (255, 0, 0)  # 已访问区域显示红色，未访问显示蓝色
                region_status = "已处理" if is_revisit else "未处理"
                
                # 显示方块中心标记和区域状态
                cv2.putText(self.frame, f"区域: {self.current_region_id} ({region_status})", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, region_color, 2)
                cv2.putText(self.frame, f"距中心: {((drone_x - center_x)**2 + (drone_y - center_y)**2)**0.5:.2f}m", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, region_color, 2)
                
                # 显示已访问区域列表
                y_offset = 150
                cv2.putText(self.frame, "已处理区域:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 20
                
                # 最多显示5个已访问区域
                visited_list = list(self.visited_regions)
                for i, region in enumerate(visited_list[:5]):
                    cv2.putText(self.frame, f"- {region}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20
                    
                if len(visited_list) > 5:
                    cv2.putText(self.frame, f"... 共{len(visited_list)}个", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20
            
            # 重置当前ROI中检测到的物体
            current_roi_detections = {}
            self.roi_animal_boxes = {}
            
            # 获取当前区域ID用于连续帧计数
            current_region = self.current_region_id
            
            # 标记是否在ROI中检测到任何动物
            detected_animal_in_roi = False

            for result in results[0].boxes:
                # huoqu jiancekuang zuobiao
                xmin = np.int64(result.xyxy[0][0].item())
                ymin = np.int64(result.xyxy[0][1].item())
                xmax = np.int64(result.xyxy[0][2].item())
                ymax = np.int64(result.xyxy[0][3].item())
                
                # 计算物体中心点坐标
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                
                # 检查目标中心是否在ROI内，如果不在则跳过处理
                is_in_roi = (self.roi_x1 <= center_x <= self.roi_x2 and 
                             self.roi_y1 <= center_y <= self.roi_y2)
                
                # 打印ROI过滤/保留信息
                if not is_in_roi:
                    rospy.loginfo(f"目标被过滤: {results[0].names[result.cls.item()]} 在位置 ({center_x}, {center_y})，不在ROI区域 [{self.roi_x1}-{self.roi_x2}, {self.roi_y1}-{self.roi_y2}] 内")
                else:
                    rospy.loginfo(f"目标保留: {results[0].names[result.cls.item()]} 在位置 ({center_x}, {center_y})，在ROI区域 [{self.roi_x1}-{self.roi_x2}, {self.roi_y1}-{self.roi_y2}] 内")
                    detected_animal_in_roi = True  # 在ROI中检测到动物
                
                # chuangjian bianjiekuang xiaoxi
                boundingBox = BoundingBox()
                boundingBox.xmin = xmin
                boundingBox.ymin = ymin
                boundingBox.xmax = xmax
                boundingBox.ymax = ymax
                boundingBox.Class = results[0].names[result.cls.item()]
                boundingBox.probability = result.conf.item()
                
                # 如果在ROI内，添加到当前检测
                if is_in_roi:
                    class_name = boundingBox.Class
                    if class_name in current_roi_detections:
                        current_roi_detections[class_name] += 1
                        # 存储动物框信息用于连续框显示
                        if class_name not in self.roi_animal_boxes:
                            self.roi_animal_boxes[class_name] = []
                        self.roi_animal_boxes[class_name].append((xmin, ymin, xmax, ymax))
                    else:
                        current_roi_detections[class_name] = 1
                        # 存储第一个框
                        self.roi_animal_boxes[class_name] = [(xmin, ymin, xmax, ymax)]
                
                # jisuan quanqiu zuobiao yongyu xianshi
                global_coords = self.calculate_global_coordinates(boundingBox)
                
                # 根据是否在ROI内使用不同颜色显示
                box_color = (0, 255, 255) if is_in_roi else (0, 255, 0)  # 黄色(ROI内)或绿色(ROI外)
                box_thickness = 3 if is_in_roi else 1
                
                # zai yuantu shang huizhi jiancekuang
                cv2.rectangle(self.frame, (int(boundingBox.xmin), int(boundingBox.ymin)), 
                             (int(boundingBox.xmax), int(boundingBox.ymax)), box_color, box_thickness)
                
                # xianshi leibie he zhixin du
                label_text = f'{boundingBox.Class}: {boundingBox.probability:.2f}'
                cv2.putText(self.frame, label_text, 
                           (int(boundingBox.xmin), int(boundingBox.ymin)-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if global_coords is not None:
                    # xianshi shishi jiesuan chulai de xy zuobiao
                    coord_text = f'XY: ({global_coords[0]:.2f}, {global_coords[1]:.2f})'
                    cv2.putText(self.frame, coord_text, 
                               (int(boundingBox.xmin), int(boundingBox.ymin)-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # panduan bingji lu mubiao suozai quyu
                    region_id = self.determine_target_region(global_coords)
                    if region_id:
                        # xianshi jiesuan chulai de zuobiao ID
                        region_text = f'ID: {region_id}'
                        cv2.putText(self.frame, region_text, 
                                   (int(boundingBox.xmin), int(boundingBox.ymin)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        # ji lu mubiao leibie he quyu ID
                        if region_id not in self.detected_regions:
                            self.detected_regions[region_id] = {}
                        
                        if boundingBox.Class not in self.detected_regions[region_id]:
                            self.detected_regions[region_id][boundingBox.Class] = 0
                        
                        self.detected_regions[region_id][boundingBox.Class] += 1
                    else:
                        # ru guo mei you qu yu ID, xian shi "No ID"
                        cv2.putText(self.frame, "ID: No ID", 
                                   (int(boundingBox.xmin), int(boundingBox.ymin)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                self.boundingBoxes.bounding_boxes.append(boundingBox)
                self.xy_pub.publish(boundingBox)
                
                # jisuan bing fabu quanqiu mubiao lichengji xinxi
                self.calculate_and_publish_global_target_odom(boundingBox)

            # 如果在当前区域ROI中检测到动物且未曾发送过扫描信号，则发布扫描信号
            if detected_animal_in_roi and self.current_region_id and self.current_region_id not in self.scan_sent_regions:
                rospy.loginfo(f"在区域 {self.current_region_id} 的ROI中检测到动物，发送扫描信号")
                self.sao_miao_pub.publish(Bool(True))
                self.scan_sent_regions.add(self.current_region_id)  # 添加到已发送区域列表
                cv2.putText(self.frame, "已发送扫描信号!", (10, 350), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 绘制ROI内的动物连续框
            self.draw_animal_continuous_boxes()
            
            # 更新连续检测计数并获取有效检测
            has_valid_detections, valid_detections = self.update_consecutive_detections(current_roi_detections)
            
            self.position_pub.publish(self.boundingBoxes)
            self.publish_image(self.frame, height, width)
            
            # 不再进行实时发布，只在区域退出时发布
            # 但仍然保留连续帧检测逻辑用于区域内动物验证
            # if has_valid_detections:
            #     self.publish_region_detection_stats(valid_detections)
            # else:
            #     # 不满足连续帧要求，不发布
            #     pass

            if self.visualize == 'True' or self.visualize is True:
                cv2.imshow('YOLOv8 Detection', self.frame)
                cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"keshihua cuowu: {str(e)}")

    def publish_image(self, imgdata, height, width):
        try:
            image_temp = Image()
            header = Header(stamp=rospy.Time.now())
            header.frame_id = self.camera_frame
            image_temp.height = height
            image_temp.width = width
            image_temp.encoding = 'bgr8'
            image_temp.data = np.array(imgdata).tobytes()
            image_temp.header = header
            image_temp.step = width * 3
            self.image_pub.publish(image_temp)
        except Exception as e:
            rospy.logerr(f"tuxiang fabu cuowu: {str(e)}")


    def draw_animal_continuous_boxes(self):
        """绘制ROI内各类动物的连续框，并显示动物信息"""
        try:
            # 定义不同动物类型的颜色
            colors = {
                'hou_zi': (0, 255, 0),      # 猴子-绿色
                'da_xiang': (255, 0, 0),    # 大象-蓝色
                'kong_que': (0, 0, 255),    # 孔雀-红色
                'lang': (255, 255, 0),      # 狼-青色
                'lao_hu': (255, 0, 255),    # 老虎-紫色
                'lion': (0, 255, 255),      # 狮子-黄色
                'monkey': (0, 255, 0),      # 猴子-绿色
                'elephant': (255, 0, 0),    # 大象-蓝色
                'peacock': (0, 0, 255),     # 孔雀-红色
                'wolf': (255, 255, 0),      # 狼-青色
                'tiger': (255, 0, 255),     # 老虎-紫色
            }
            
            # 遍历ROI中的各类动物
            y_offset = 150  # 文字显示的起始y坐标
            for animal_type, boxes in self.roi_animal_boxes.items():
                if not boxes:
                    continue
                    
                # 获取该类动物的颜色
                color = colors.get(animal_type, (0, 255, 255))  # 默认黄色
                
                # 获取边界框数量
                box_count = len(boxes)
                
                # 绘制动物类型和计数信息
                info_text = f"{animal_type}: {box_count} 只"
                cv2.putText(self.frame, info_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30
                
                # 绘制该类动物的所有边界框
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    # 绘制边界框
                    cv2.rectangle(self.frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    
                    # 在框上方显示类型名称
                    cv2.putText(self.frame, animal_type, (int(xmin), int(ymin)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        except Exception as e:
            rospy.logerr(f"绘制动物连续框错误: {str(e)}")
            
    def update_consecutive_detections(self, roi_detections):
        """更新连续识别帧计数并记录区域内的动物"""
        try:
            if self.current_region_id is None:
                return False, {}
            
            # 检查是否在已访问过的区域内，如果是则直接返回空结果
            if self.current_region_id in self.visited_regions:
                rospy.loginfo(f"区域 {self.current_region_id} 已被访问过，忽略新的动物检测")
                return False, {}
            
            # 初始化区域动物记录
            if self.current_region_id not in self.region_animals:
                self.region_animals[self.current_region_id] = {}
                
            if self.current_region_id not in self.validated_animals:
                self.validated_animals[self.current_region_id] = {}
                
            # 初始化当前区域的连续检测记录
            if self.current_region_id not in self.consecutive_detections:
                self.consecutive_detections[self.current_region_id] = {}
            
            # 更新连续检测记录
            for animal_type, count in roi_detections.items():
                # 更新连续帧计数
                if animal_type not in self.consecutive_detections[self.current_region_id]:
                    # 首次检测到该动物
                    self.consecutive_detections[self.current_region_id][animal_type] = 1
                else:
                    # 增加连续检测帧数
                    self.consecutive_detections[self.current_region_id][animal_type] += 1
                
                # 检查是否达到了区域内所需帧数（6帧）
                frame_count = self.consecutive_detections[self.current_region_id][animal_type]
                if frame_count >= self.required_region_frames:
                    # 将该动物标记为已验证
                    self.validated_animals[self.current_region_id][animal_type] = True
                    
                    # 更新区域内动物计数
                    if animal_type not in self.region_animals[self.current_region_id]:
                        self.region_animals[self.current_region_id][animal_type] = count
                    else:
                        # 如果已经有记录，保留最大值
                        self.region_animals[self.current_region_id][animal_type] = max(
                            self.region_animals[self.current_region_id][animal_type],
                            count
                        )
                    
            # 检查当前帧未检测到的动物，重置它们的连续帧计数
            for animal_type in list(self.consecutive_detections[self.current_region_id].keys()):
                if animal_type not in roi_detections:
                    self.consecutive_detections[self.current_region_id][animal_type] = 0
            
            # 检查是否有动物达到了连续识别阈值（用于旧的实时发送逻辑）
            valid_detections = {}
            for animal_type, frame_count in self.consecutive_detections[self.current_region_id].items():
                if frame_count >= self.required_consecutive_frames:
                    # 动物连续检测帧数达到阈值，算作有效检测
                    valid_detections[animal_type] = roi_detections.get(animal_type, 0)
                    
            # 显示连续帧计数信息
            y_offset = 250  # 文字显示的起始y坐标
            cv2.putText(self.frame, f"连续帧要求: {self.required_consecutive_frames} (实时) / {self.required_region_frames} (区域)", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            for animal_type, frame_count in self.consecutive_detections[self.current_region_id].items():
                # 显示两个状态：实时发送状态和区域记录状态
                realtime_status = "有效" if frame_count >= self.required_consecutive_frames else "等待"
                region_status = "已记录" if animal_type in self.validated_animals.get(self.current_region_id, {}) else "记录中"
                
                # 显示实时状态
                realtime_color = (0, 255, 0) if frame_count >= self.required_consecutive_frames else (0, 0, 255)
                cv2.putText(self.frame, f"{animal_type}: {frame_count}/{self.required_consecutive_frames} ({realtime_status})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, realtime_color, 1)
                y_offset += 20
                
                # 显示区域记录状态
                region_color = (255, 255, 0) if animal_type in self.validated_animals.get(self.current_region_id, {}) else (0, 165, 255)
                cv2.putText(self.frame, f" └─ 区域记录: {frame_count}/{self.required_region_frames} ({region_status})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)
                y_offset += 20
                
            # 显示区域内累积的动物信息
            if self.current_region_id in self.region_animals and self.region_animals[self.current_region_id]:
                cv2.putText(self.frame, f"区域 {self.current_region_id} 记录的动物:", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                
                for animal_type, count in self.region_animals[self.current_region_id].items():
                    cv2.putText(self.frame, f" - {animal_type}: {count}只", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 20
            
            # 返回实时发送逻辑的结果
            return len(valid_detections) > 0, valid_detections
            
        except Exception as e:
            rospy.logerr(f"更新连续帧计数错误: {str(e)}")
            return False, {}


    def determine_current_region(self):
        """根据无人机当前位置确定所在区域ID（使用统一坐标系：B1 A9为起点，x朝前，y朝左）"""
        try:
            if self.current_drone_odom is None:
                return None
                
            # 获取无人机位置
            drone_x = self.drone_position[0]
            drone_y = self.drone_position[1]
            
            # 使用统一坐标系：B1 A9方块中心为起点(0,0)，x朝前，y朝左
            # 方格大小为0.5m
            cell_size = 0.5
            
            # 计算网格坐标
            # x朝前对应B行增加，y朝左对应A列减少
            grid_row = int(round(drone_x / cell_size))  # B1=0, B2=1, B3=2, ..., B7=6
            grid_col = int(round(drone_y / cell_size))  # A9=0, A8=1, A7=2, ..., A1=8
            
            # 边界检查
            if grid_row < 0 or grid_row >= 7 or grid_col < 0 or grid_col >= 9:
                return None
                
            # 计算方格中心的全局坐标
            center_global_x = grid_row * cell_size
            center_global_y = grid_col * cell_size
            
            # 计算到方格中心的距离
            distance = math.sqrt((drone_x - center_global_x)**2 + (drone_y - center_global_y)**2)
            
            # 如果在容差范围内，返回区域ID
            if distance <= self.region_tolerance:
                # 转换为区域ID格式：B{row+1} A{9-col}
                b_index = grid_row + 1  # B1, B2, ..., B7
                a_index = 9 - grid_col  # A9, A8, ..., A1
                
                region_id = f"B{b_index} A{a_index}"
                return region_id
                
            return None
            
        except Exception as e:
            rospy.logerr(f"判断当前区域错误: {str(e)}")
            return None
    
    def publish_region_detection_stats(self, roi_detections):
        """【已弃用】不再使用实时发布模式，仅在区域退出时发布数据"""
        # 此函数已不再被调用，保留仅作参考
        rospy.logwarn("publish_region_detection_stats函数已弃用，系统现在只在区域退出时发布数据")
        try:
            # 只有当无人机在某个区域内时才发布
            if self.current_region_id is None:
                return
                
            # 位置验证已经在determine_current_region中完成
            # 如果current_region_id不为空，说明已经在某个有效区域内
            # 不再需要额外的距离检查，直接发布数据
            if self.current_region_id is None:
                rospy.loginfo("无人机不在任何有效区域内，不发布实时数据")
                return
                
            # 调试信息：打印无人机坐标和区域ID
            drone_x, drone_y, _ = self.drone_position
            rospy.loginfo(f"无人机当前位置: ({drone_x:.2f}, {drone_y:.2f})，区域ID: {self.current_region_id}")
                
            # 使用传入的ROI内的动物检测结果
            class_counts = roi_detections
            
            # 检查是否有任何检测
            if not class_counts:
                rospy.loginfo("ROI区域内无动物检测，不发布实时数据")
                return
            
            # 检查该区域是否已经识别过（只适用于旧的实时发送逻辑）
            if self.current_region_id in self.recognized_regions:
                rospy.loginfo(f"区域 {self.current_region_id} 已经实时识别过，不再发送实时数据")
                return
            
            # 构建JSON格式的消息（只包含当前区域内的检测结果）
            detection_data = {
                "region_id": self.current_region_id,
                "detections": class_counts,
                "mode": "realtime"  # 标记为实时检测
            }
            
            # 转换为JSON字符串
            detection_msg = json.dumps(detection_data, ensure_ascii=False)
            
            # 发布检测结果到两个话题
            self.region_detection_pub.publish(String(detection_msg))
            self.id_dongwu_value_pub.publish(String(detection_msg))
            
            # 记录已识别过的区域（只用于实时发送逻辑）
            self.recognized_regions.add(self.current_region_id)
            
            # 输出详细日志
            rospy.loginfo(f"实时模式: 发布区域 {self.current_region_id} 动物检测数据: {detection_msg}")
            
            # 打印连续框信息
            for animal_type, boxes in self.roi_animal_boxes.items():
                rospy.loginfo(f"检测到 {animal_type}: {len(boxes)}只")
            
            # 额外的调试信息
            roi_detection_count = sum(class_counts.values())
            rospy.loginfo(f"在区域 {self.current_region_id} 的ROI内总共检测到 {roi_detection_count} 个动物")
            
        except Exception as e:
            rospy.logerr(f"发布区域检测统计错误: {str(e)}")
    

    def calculate_global_coordinates(self, boundingBox):
        """jisuan mubiao de quanqiu zuobiao"""
        if self.current_drone_odom is None:
            return None
        
        # jisuan mubiao zai tuxiang zhong de zhongxin dian
        center_x = (boundingBox.xmin + boundingBox.xmax) / 2.0
        center_y = (boundingBox.ymin + boundingBox.ymax) / 2.0
        
        # shiyong xiangji neicanhe wurenji gaodu jisuan mubiao zai wurenji zuobiaoxia de pianyi
        cx_actual = self.cx
        cy_actual = self.cy
        
        # jisuan mubiao xiangdui yu xiangji zhongxin de pianyi (xiangsu)
        pixel_offset_x = center_x - cx_actual
        pixel_offset_y = center_y - cy_actual
        
        # huoqu wurenji dangqian de zitai (sixiangshu)
        orientation = self.current_drone_odom.pose.pose.orientation
        # zhuanhuan wei yaogunzhongxin - zuo you - shangxia (roll-pitch-yaw)
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # shiyong xiangsi sanjiaoxing yuanli jisuan shiji pianyi (mi)
        # bing kaolv wurenji qingxie jiaozheng (roll & pitch)
        
        # jisuan xiangji de z zhou gaodu (kaolv wurenji qingxie)
        # jiaoju fangxiang chengxiang shi fuyou zhengfu, suo yi shi angle_correction = cos(roll)*cos(pitch)
        angle_correction = math.cos(roll) * math.cos(pitch)
        corrected_height = self.drone_height / angle_correction if angle_correction > 0.1 else self.drone_height
        
        # jiaozheng xiangsu pianyi (kaolv pitch he roll)
        # pitch yingxiang y zhou (qianhou)
        # roll yingxiang x zhou (zuo you)
        corrected_pixel_offset_y = pixel_offset_y - math.tan(pitch) * self.fy
        corrected_pixel_offset_x = pixel_offset_x - math.tan(roll) * self.fx
        
        # shiyong jiaozheng hou de gaodu he xiangsu pianyi jisuan shiji pianyi
        # zhu: xiangji zuobiaxi zhong, x xiangqian (chengxiang fangxiang), y xiangyou
        # dan wurenji zuobiaxi zhong, x chaoqian, y chaozuo
        real_offset_forward = -corrected_pixel_offset_y * corrected_height / self.fy
        real_offset_left = -corrected_pixel_offset_x * corrected_height / self.fx
        
        # huoqu wurenji dangqian weizhi he zitai
        drone_x = self.current_drone_odom.pose.pose.position.x
        drone_y = self.current_drone_odom.pose.pose.position.y
        
        # jiang mubiao pianyi zhuanhuan dao quanqiu zuobiaxi (x chaoqian, y chaozuo)
        global_offset_x = real_offset_forward * math.cos(yaw) - real_offset_left * math.sin(yaw)
        global_offset_y = real_offset_forward * math.sin(yaw) + real_offset_left * math.cos(yaw)
        
        # jisuan mubiao de quanqiu weizhi
        # gen ju yonghu fanku, xu yao xiuzheng zuobiaxi pianyi
        # mubiao x=1m, y=1.5m shi, ying gai xianshi x=0.5m, y=0.5m
        # suo yi xu yao jian qu 0.5m de pianyi
        target_global_x = drone_x + global_offset_x - 0.5
        target_global_y = drone_y + global_offset_y - 1
        
        return (target_global_x, target_global_y)
    
    def calculate_and_publish_global_target_odom(self, boundingBox):
        """jisuan bing fabu quanqiu mubiao lichengji xinxi"""
        if self.current_drone_odom is None:
            rospy.logwarn("No odometry data received yet")
            return
        
        # jisuan mubiao zai tuxiang zhong de zhongxin dian
        center_x = (boundingBox.xmin + boundingBox.xmax) / 2.0
        center_y = (boundingBox.ymin + boundingBox.ymax) / 2.0
        
        # shiyong xiangji neicanhe wurenji gaodu jisuan mubiao zai wurenji zuobiaoxia de pianyi
        # xiangji neicanzhong de zhongxin dian (cx, cy) shi xiangdui yu yuantu de
        cx_actual = self.cx
        cy_actual = self.cy
        
        # jisuan mubiao xiangdui yu xiangji zhongxin de pianyi (xiangsu)
        pixel_offset_x = center_x - cx_actual
        pixel_offset_y = center_y - cy_actual
        
        # huoqu wurenji dangqian de zitai (sixiangshu)
        orientation = self.current_drone_odom.pose.pose.orientation
        # zhuanhuan wei yaogunzhongxin - zuo you - shangxia (roll-pitch-yaw)
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # shiyong xiangsi sanjiaoxing yuanli jisuan shiji pianyi (mi)
        # bing kaolv wurenji qingxie jiaozheng (roll & pitch)
        
        # jisuan xiangji de z zhou gaodu (kaolv wurenji qingxie)
        # jiaoju fangxiang chengxiang shi fuyou zhengfu, suo yi shi angle_correction = cos(roll)*cos(pitch)
        angle_correction = math.cos(roll) * math.cos(pitch)
        corrected_height = self.drone_height / angle_correction if angle_correction > 0.1 else self.drone_height
        
        # jiaozheng xiangsu pianyi (kaolv pitch he roll)
        # pitch yingxiang y zhou (qianhou)
        # roll yingxiang x zhou (zuo you)
        corrected_pixel_offset_y = pixel_offset_y - math.tan(pitch) * self.fy
        corrected_pixel_offset_x = pixel_offset_x - math.tan(roll) * self.fx
        
        # shiyong jiaozheng hou de gaodu he xiangsu pianyi jisuan shiji pianyi
        # zhu: xiangji zuobiaxi zhong, x xiangqian (chengxiang fangxiang), y xiangyou
        # dan wurenji zuobiaxi zhong, x chaoqian, y chaozuo
        real_offset_forward = -corrected_pixel_offset_y * corrected_height / self.fy  # xiangji -y -> wurenji x (qianhou)
        real_offset_left = -corrected_pixel_offset_x * corrected_height / self.fx     # xiangji -x -> wurenji y (zuoyou)
        
        # huoqu wurenji dangqian weizhi he zitai
        drone_x = self.current_drone_odom.pose.pose.position.x
        drone_y = self.current_drone_odom.pose.pose.position.y
        drone_z = self.current_drone_odom.pose.pose.position.z
        
        # jiang mubiao pianyi zhuanhuan dao quanqiu zuobiaxi (x chaoqian, y chaozuo)
        # kaolv wurenji de yaw jiao jinxing zuobiaxi xuanzhuan
        global_offset_x = real_offset_forward * math.cos(yaw) - real_offset_left * math.sin(yaw)
        global_offset_y = real_offset_forward * math.sin(yaw) + real_offset_left * math.cos(yaw)
        
        # jisuan mubiao de quanqiu weizhi
        target_global_x = drone_x + global_offset_x
        target_global_y = drone_y + global_offset_y
        target_global_z = 0.0  # jiashe mubiao zai dimian shang
        
        # jisuan mubiao xiangdui yu wurenji de fangxiang jiao
        target_yaw = math.atan2(global_offset_y, global_offset_x)
        target_quaternion = tf.transformations.quaternion_from_euler(0, 0, target_yaw)
        
        # chuangjian bing fabu Odometry xiaoxi
        target_odom = Odometry()
        target_odom.header.stamp = rospy.Time.now()
        target_odom.header.frame_id = "map"  # quanqiu zuobiaxi
        target_odom.child_frame_id = "target"
        
        # shezhimubiao weizhi
        target_odom.pose.pose.position.x = target_global_x
        target_odom.pose.pose.position.y = target_global_y
        target_odom.pose.pose.position.z = target_global_z
        
        # shezhi mubiao zitai (sishu shu)
        target_odom.pose.pose.orientation.x = target_quaternion[0]
        target_odom.pose.pose.orientation.y = target_quaternion[1]
        target_odom.pose.pose.orientation.z = target_quaternion[2]
        target_odom.pose.pose.orientation.w = target_quaternion[3]
        
        # shezhi xiefang cha juzhen (jiashe mubiao jingzhi)
        target_odom.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                      0, 0.1, 0, 0, 0, 0,
                                      0, 0, 0.1, 0, 0, 0,
                                      0, 0, 0, 0.1, 0, 0,
                                      0, 0, 0, 0, 0.1, 0,
                                      0, 0, 0, 0, 0, 0.1]
        
        # shezhi sudu wei ling (jiashe mubiao jingzhi)
        target_odom.twist.twist.linear.x = 0.0
        target_odom.twist.twist.linear.y = 0.0
        target_odom.twist.twist.linear.z = 0.0
        target_odom.twist.twist.angular.x = 0.0
        target_odom.twist.twist.angular.y = 0.0
        target_odom.twist.twist.angular.z = 0.0
        
        # shezhi sudu xiefang cha juzhen
        target_odom.twist.covariance = [0.1, 0, 0, 0, 0, 0,
                                       0, 0.1, 0, 0, 0, 0,
                                       0, 0, 0.1, 0, 0, 0,
                                       0, 0, 0, 0, 0.1, 0,
                                       0, 0, 0, 0, 0.1, 0,
                                       0, 0, 0, 0, 0, 0.1]
        
        # fabu mubiao lichengji xinxi
        self.global_target_odom_pub.publish(target_odom)

    def determine_target_region(self, global_coords):
        """根据目标全局坐标确定所在区域ID（使用统一坐标系：B1 A9为起点，x朝前，y朝左）"""
        try:
            if global_coords is None or self.current_drone_odom is None:
                return None
            
            target_x, target_y = global_coords
            
            # 使用统一坐标系：B1 A9方块中心为起点(0,0)，x朝前，y朝左
            # 方格大小为0.5m
            cell_size = 0.5
            
            # 计算网格坐标
            # x朝前对应B行增加，y朝左对应A列减少
            grid_row = int(round(target_x / cell_size))  # B1=0, B2=1, B3=2, ..., B7=6
            grid_col = int(round(target_y / cell_size))  # A9=0, A8=1, A7=2, ..., A1=8
            
            # 边界检查
            if grid_row < 0 or grid_row >= 7 or grid_col < 0 or grid_col >= 9:
                return None
                
            # 计算方格中心的全局坐标
            center_global_x = grid_row * cell_size
            center_global_y = grid_col * cell_size
            
            # 计算到方格中心的距离
            distance = math.sqrt((target_x - center_global_x)**2 + (target_y - center_global_y)**2)
            
            # 如果在容差范围内，返回区域ID
            if distance <= self.region_tolerance:
                # 转换为区域ID格式：B{row+1} A{9-col}
                b_index = grid_row + 1  # B1, B2, ..., B7
                a_index = 9 - grid_col  # A9, A8, ..., A1
                
                region_id = f"B{b_index} A{a_index}"
                return region_id
                
            return None
            
        except Exception as e:
            rospy.logerr(f"判断目标区域错误: {str(e)}")
            return None

    def global_to_grid_coords(self, global_x, global_y):
        """将全局坐标转换为网格坐标"""
        try:
            # 使用统一坐标系：B1 A9方块中心为起点(0,0)，x朝前，y朝左
            # 方格大小为0.5m
            cell_size = 0.5
            
            # 计算网格坐标
            # x朝前对应B行增加，y朝左对应A列减少
            grid_row = int(round(global_x / cell_size))  # B1=0, B2=1, B3=2, ..., B7=6
            grid_col = int(round(global_y / cell_size))  # A9=0, A8=1, A7=2, ..., A1=8
            
            return grid_col, grid_row
        except Exception as e:
            rospy.logerr(f"全局坐标转网格坐标错误: {str(e)}")
            return None, None


    def __del__(self):
        cv2.destroyAllWindows()

def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    try:
        yolo_dect = Yolo_Dect()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
