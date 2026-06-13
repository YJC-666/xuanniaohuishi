#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
喷涂进度检测与覆盖率计算模块

实时采集喷涂画面，与设计图对比，计算喷涂覆盖率和质量。
替代原有的 YOLO 动物检测功能。

话题:
  发布: /mcp/coverage  (JSON: {coverage, quality, timestamp})
  订阅: /camera/color/image_raw
"""

import cv2
import numpy as np
import rospy
import json
import math
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SprayCoverageDetector:
    """喷涂覆盖率检测器"""

    def __init__(self):
        self.bridge = CvBridge()

        # 参数
        self.design_image = None          # 设计图 (作为参考)
        self.design_binary = None         # 设计图二值化
        self.image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.pub_topic = rospy.get_param('~pub_topic', '/mcp/coverage')
        self.visualize = rospy.get_param('~visualize', True)

        # 喷涂检测参数
        self.spray_color_lower = np.array([80, 30, 30])    # HSV 下限 (白色/浅色喷涂)
        self.spray_color_upper = np.array([180, 80, 255])  # HSV 上限
        self.coverage_threshold = 0.3     # 像素被认为是"已喷涂"的亮度阈值

        # 状态
        self.latest_frame = None
        self.coverage = 0.0
        self.quality_score = 0.0
        self.frame_count = 0

        # 订阅与发布
        self.img_sub = rospy.Subscriber(
            self.image_topic, Image, self._image_callback, queue_size=1)
        self.coverage_pub = rospy.Publisher(
            self.pub_topic, String, queue_size=10)
        self.debug_pub = rospy.Publisher(
            '/mcp/debug_image', Image, queue_size=1)

        rospy.loginfo("[Coverage] 喷涂覆盖率检测器已初始化")
        rospy.loginfo(f"          图像话题: {self.image_topic}")
        rospy.loginfo(f"          发布话题: {self.pub_topic}")

    def load_design(self, image_path):
        """加载设计图作为参考"""
        img = cv2.imread(image_path)
        if img is None:
            rospy.logerr(f"[Coverage] 无法加载参考图: {image_path}")
            return False

        self.design_image = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.design_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        rospy.loginfo(f"[Coverage] 加载参考图: {image_path} "
                      f"({img.shape[1]}x{img.shape[0]})")
        return True

    def set_design_from_bytes(self, img_bytes, width, height):
        """从字节数据加载参考图（来自航点规划器）"""
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 3)
        self.design_image = arr.copy()
        gray = cv2.cvtColor(self.design_image, cv2.COLOR_BGR2GRAY)
        _, self.design_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def _image_callback(self, msg):
        """处理相机帧"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_frame = cv_img
            self.frame_count += 1

            # 每 5 帧计算一次覆盖率
            if self.frame_count % 5 == 0 and self.design_binary is not None:
                self._compute_coverage(cv_img)
        except Exception as e:
            rospy.logerr(f"[Coverage] 图像处理错误: {e}")

    def _compute_coverage(self, frame):
        """计算喷涂覆盖率

        通过比较当前帧中喷涂区域与设计图目标区域的像素重叠来计算。
        """
        try:
            # 1. 将帧缩放到与设计图相同尺寸
            if frame.shape[:2] != self.design_image.shape[:2]:
                frame = cv2.resize(
                    frame,
                    (self.design_image.shape[1], self.design_image.shape[0])
                )

            # 2. 检测喷涂区域
            #    方法: 将帧转为灰度，检测高亮度区域（喷涂颜料通常比背景亮）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 自适应阈值提取"已喷涂"像素
            _, sprayed_mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 3. 计算覆盖率
            #    覆盖率 = (已喷涂 & 设计图目标区域) 的像素 / 设计图目标区域总像素
            design_pixels = cv2.countNonZero(self.design_binary)
            if design_pixels == 0:
                self.coverage = 0.0
            else:
                overlap = cv2.bitwise_and(sprayed_mask, self.design_binary)
                sprayed_pixels = cv2.countNonZero(overlap)
                self.coverage = sprayed_pixels / design_pixels

            # 4. 计算喷涂质量
            #    质量评分基于喷涂区域的边缘清晰度
            self.quality_score = self._compute_quality(frame, sprayed_mask)

            # 5. 发布结果
            self._publish_result()

            # 6. 可视化
            if self.visualize:
                self._visualize(frame, sprayed_mask, overlap)

        except Exception as e:
            rospy.logerr(f"[Coverage] 覆盖率计算错误: {e}")

    def _compute_quality(self, frame, mask):
        """计算喷涂质量评分 (0-1)"""
        try:
            # 使用边缘检测评估喷涂边缘清晰度
            edges = cv2.Canny(frame, 50, 150)
            mask_edges = cv2.bitwise_and(edges, mask)

            edge_pixels = cv2.countNonZero(mask_edges)
            total_pixels = cv2.countNonZero(mask)

            if total_pixels == 0:
                return 0.0

            # 边缘越多说明喷涂边界越清晰
            edge_ratio = edge_pixels / total_pixels
            # 归一化到 0-1 (通常边缘比例在 0.01-0.1 之间)
            quality = min(1.0, edge_ratio * 20)
            return quality

        except Exception:
            return 0.0

    def _publish_result(self):
        """发布覆盖率和质量数据"""
        msg = json.dumps({
            'coverage': round(self.coverage * 100, 1),
            'quality': round(self.quality_score * 100, 1),
            'timestamp': rospy.Time.now().to_sec()
        })
        self.coverage_pub.publish(String(msg))

    def _visualize(self, frame, mask, overlap):
        """调试可视化"""
        vis = frame.copy()

        # 叠加覆盖率信息
        cv2.putText(vis, f"Coverage: {self.coverage*100:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Quality: {self.quality_score*100:.1f}%",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 200, 0), 2)

        # 显示对比图
        h, w = frame.shape[:2]
        debug = np.zeros((h, w*2, 3), dtype=np.uint8)
        debug[:, :w] = vis

        # 右侧显示覆盖热力图
        overlay = cv2.cvtColor(overlap, cv2.COLOR_GRAY2BGR)
        overlay[:, :, 2] = overlap  # 红色通道显示已覆盖区域
        debug[:, w:] = overlay

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception:
            pass


def main():
    rospy.init_node('spray_coverage_detector', anonymous=True)
    detector = SprayCoverageDetector()

    # 检查是否提供了设计图参数
    design_path = rospy.get_param('~design_path', '')
    if design_path:
        detector.load_design(design_path)

    rospy.spin()


if __name__ == '__main__':
    main()