#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
from std_msgs.msg import Int32, String
from nav_msgs.msg import Odometry
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import yaml
import math
import os
from datetime import datetime

class WildlifeSurveyStation(QMainWindow):
    # 添加信号定义
    update_position_signal = pyqtSignal(list)
    update_status_signal = pyqtSignal(str, int)
    update_wildlife_display_signal = pyqtSignal()
    update_map_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # 初始化ROS节点
        rospy.init_node('wildlife_survey_station', anonymous=True)
        
        # 创建发布者
        self.command_pub = rospy.Publisher('/mission_command', Int32, queue_size=10)
        
        # 网格参数 - 63个50cm×50cm方格
        # 行：B1-B7（7行，B1在底部，B7在顶部）
        # 列：A1-A9（9列，A1在左侧，A9在右侧）
        # 总计：9×7 = 63个方格，每个方格50cm×50cm
        self.grid_rows = 7  # 行数 (B1-B7，共7行)
        self.grid_cols = 9  # 列数 (A1-A9，共9列)
        self.cell_size = 0.5  # 每个方格边长50cm = 0.5米
        
        # 红点位置 (起降点) - 设置在B1 A9位置
        self.red_point = (8, 6)  # B1 A9位置 (A9列B1行，即右下角)

        # 全局坐标系设置：以红点为原点，X向前(B1->B7)，Y向左(A9->A1)
        self.origin_col, self.origin_row = self.red_point  # 原点位置设置为红点位置

        # 高度设置
        self.takeoff_landing_height = 0.0  # 起降高度
        self.survey_height = 1.22  # 巡查高度
        
        # 禁区列表
        self.forbidden_zones = []
        
        # 航点列表
        self.waypoints = []
        
        # 返回路径起始索引（用于区分巡查路径和返回路径）
        self.return_path_start_index = -1
        
        # 任务状态
        self.mission_active = False
        
        # 任务完成标志
        self.mission_completed = False
        
        # 无人机是否回到原点的标志
        self.drone_returned_to_origin = False
        
        # 回到原点的检测距离阈值(米)
        self.return_to_origin_threshold = 0.3
        
        # 上一次保存数据的时间
        self.last_save_time = 0
        
        # 定时器用于发布命令
        self.timer = QTimer()
        self.timer.timeout.connect(self.publish_command)
        self.timer.start(200)  # 5Hz

        # ROS发布器 - 用于发布航点数据
        self.waypoint_publisher = rospy.Publisher('/wildlife_survey/waypoints', String, queue_size=10)
        
        # 无人机当前位置
        self.drone_position = [0, 0, 0]  # [x, y, z] 单位：米
        
        # 无人机初始位置（用于坐标系初始化）
        self.drone_initial_position = None
        
        # 是否已初始化坐标系
        self.coordinate_system_initialized = False
        
        # 坐标系偏移量（从无人机初始位置到红点的偏移）
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        
        # 无人机轨迹记录（不限制轨迹点数量）
        self.drone_trajectory = []
        
        # 无人机上一次显示的网格位置
        self.last_displayed_grid_position = None
        
        # 动物检测记录 - 用于存储每个方格检测到的无人机喷涂信息
        # 格式: {(grid_col, grid_row): {'detection_time': timestamp, 'animals': {'种类1': 数量1, '种类2': 数量2, ...}}}
        self.wildlife_detections = {}
        
        # 任务ID - 用于保存历史记录
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 历史记录目录
        self.history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yesheng_log')
        
        # 确保历史记录目录存在
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            rospy.loginfo(f"创建历史记录目录: {self.history_dir}")
        
        # 已加载的历史记录ID
        self.loaded_history_id = None
        
        # 历史记录列表
        self.history_records = self.load_history_list()
        
        # 历史面板是否显示
        self.history_panel_visible = False
        
        # 当前方格位置 - 用于判断是否进入新方格
        self.current_grid_position = None
        
        # 上次检测时间 - 避免同一方格短时间内重复检测
        self.last_detection_time = 0
        
        # 动物种类中英文映射
        self.animal_name_map = {
            'hou_zi': '猴子',
            'da_xiang': '大象',
            'kong_que': '孔雀',
            'lang': '狼',
            'lao_hu': '老虎',
            'lion': '狮子',
            'monkey': '猴子',
            'elephant': '大象',
            'peacock': '孔雀',
            'wolf': '狼',
            'tiger': '老虎'
        }

        # 连接信号和槽
        self.update_position_signal.connect(self._handle_position_update)
        self.update_status_signal.connect(self.statusBar().showMessage)
        self.update_wildlife_display_signal.connect(self._update_wildlife_display_safe)
        self.update_map_signal.connect(self._update_map_safe)
        
        # 创建订阅器 - 订阅无人机位置
        rospy.Subscriber('iris_0/mavros/local_position/odom', Odometry, self.odom_callback, queue_size=10)
        
        # 创建订阅器 - 订阅目标检测结果
        rospy.Subscriber('/roi_detection_stats', String, self.detection_callback, queue_size=10)
        
        # 定时器用于更新无人机位置显示（5Hz）
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_drone_display)
        self.position_timer.start(200)  # 5Hz (200ms)
        
        self.init_ui()

        # 测试坐标转换（可选，用于验证）
        self.test_coordinate_conversion()
        
    def _handle_position_update(self, position):
        """处理位置更新的槽函数（运行在UI线程）"""
        self.drone_position = position
        
        # 检查是否需要初始化坐标系
        if self.drone_initial_position is None:
            self.drone_initial_position = position[:]
            rospy.loginfo(f"记录无人机初始位置: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            self.initialize_coordinate_system()
        
        # 检测无人机是否回到原点（红点位置）
        if self.mission_active and not self.mission_completed:
            # 计算无人机到原点的距离
            drone_x, drone_y, drone_z = position
            distance_to_origin = math.sqrt(drone_x**2 + drone_y**2)
            height_near_ground = abs(drone_z) < 0.1  # 高度接近地面
            
            # 检查无人机是否靠近原点并且已经接近地面（可能已着陆）
            if distance_to_origin < self.return_to_origin_threshold and height_near_ground:
                if not self.drone_returned_to_origin:
                    self.drone_returned_to_origin = True
                    rospy.loginfo(f"无人机已回到原点，距离: {distance_to_origin:.2f}米，高度: {drone_z:.2f}米")
                    
                    # 任务完成，自动保存无人机喷涂数据
                    current_time = rospy.Time.now().to_sec()
                    # 避免短时间内重复保存
                    if current_time - self.last_save_time > 5.0 and self.wildlife_detections:
                        self.save_wildlife_data()
                        self.last_save_time = current_time
                        self.statusBar().showMessage("任务完成，已自动保存无人机喷涂数据", 5000)
                        self.mission_completed = True
                        
                        # 发送停止命令
                        command = Int32()
                        command.data = 0  # 0表示停止任务
                        self.command_pub.publish(command)
            else:
                # 如果无人机已经离开原点，重置标志，以便下次可以再次检测返回原点事件
                self.drone_returned_to_origin = False
    
    def _update_wildlife_display_safe(self):
        """线程安全的更新无人机喷涂显示（运行在UI线程）"""
        self.update_wildlife_display()
    
    def _update_map_safe(self):
        """线程安全的更新地图（运行在UI线程）"""
        self.map_widget.update()
    
    def odom_callback(self, msg):
        """处理接收到的里程计数据（运行在ROS回调线程）"""
        # 获取无人机位置（x, y, z）
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # 通过信号发送位置更新
        self.update_position_signal.emit([x, y, z])
        
    def detection_callback(self, msg):
        """处理动物检测结果回调（运行在ROS回调线程）"""
        try:
            # 如果正在查看历史记录，不处理新的检测结果
            if self.loaded_history_id is not None:
                return
                
            # 获取当前时间
            current_time = rospy.Time.now().to_sec()
            
            # 获取当前网格位置
            drone_x, drone_y, _ = self.drone_position
            grid_col, grid_row = self.global_to_grid_coords(drone_x, drone_y)
            current_grid = (grid_col, grid_row)
            
            # 检查是否符合检测条件:
            # 1. 无人机足够接近方格中心(小于0.1米)
            # 2. 进入新方格或距离上次检测时间超过3秒
            # 3. 当前方格之前未检测过
            if (hasattr(self, 'last_distance_to_center') and 
                self.last_distance_to_center < 0.1 and
                (current_grid != self.current_grid_position or 
                 current_time - self.last_detection_time > 3.0) and
                current_grid not in self.wildlife_detections):
                
                # 更新当前方格位置和检测时间
                self.current_grid_position = current_grid
                self.last_detection_time = current_time
                
                # 解析检测消息
                detection_data = msg.data.strip()
                if detection_data and detection_data != "wu_jiance_mubiao":
                    # 消息格式: "class1:count1 class2:count2 ..."
                    detections = {}
                    for item in detection_data.split():
                        parts = item.split(':')
                        if len(parts) == 2:
                            animal_type, count = parts
                            # 将拼音转换为中文（如果映射中有的话）
                            animal_name = self.animal_name_map.get(animal_type, animal_type)
                            detections[animal_name] = int(count)
                    
                    # 如果有检测到动物，保存到检测记录中
                    if detections:
                        self.wildlife_detections[current_grid] = {
                            'detection_time': current_time,
                            'animals': detections,
                            'grid_coord': self.position_to_coord(grid_col, grid_row)
                        }
                        
                        # 在状态栏显示信息（通过信号）
                        animal_info = ", ".join([f"{name}: {count}只" for name, count in detections.items()])
                        grid_name = self.wildlife_detections[current_grid]['grid_coord']
                        status_msg = f"在{grid_name}检测到: {animal_info}"
                        self.update_status_signal.emit(status_msg, 5000)
                        
                        # 更新动物信息显示区域（通过信号）
                        self.update_wildlife_display_signal.emit()
                        
                        # 更新地图显示（通过信号）
                        self.update_map_signal.emit()
            
        except Exception as e:
            rospy.logerr(f"处理检测数据出错: {str(e)}")
    
    def initialize_coordinate_system(self):
        """初始化坐标系，使无人机初始位置对应于红点位置"""
        if self.drone_initial_position is None:
            rospy.logwarn("无法初始化坐标系：无人机初始位置未知")
            return
            
        rospy.loginfo("初始化坐标系，使无人机当前位置对应于红点位置")
        
        # 坐标系偏移量就是无人机初始位置的负值（这样无人机初始位置在转换后为(0,0,0)）
        self.offset_x = -self.drone_initial_position[0]
        self.offset_y = -self.drone_initial_position[1] 
        self.offset_z = -self.drone_initial_position[2]
        
        rospy.loginfo(f"坐标系偏移量: X={self.offset_x:.2f}, Y={self.offset_y:.2f}, Z={self.offset_z:.2f}")
        
        # 标记坐标系已初始化
        self.coordinate_system_initialized = True
        
    def update_drone_display(self):
        """更新无人机位置显示"""
        # 计算无人机当前位置对应的网格坐标
        drone_x, drone_y, drone_z = self.drone_position
        grid_col, grid_row = self.global_to_grid_coords(drone_x, drone_y)
        
        # 如果坐标系已初始化，显示相对于初始位置的偏移
        if self.coordinate_system_initialized:
            # 显示更详细的位置信息
            self.drone_position_label.setText(
                f"无人机位置: X={drone_x:.2f}m Y={drone_y:.2f}m Z={drone_z:.2f}m " +
                f"(相对初始点: X={(drone_x+self.offset_x):.2f}m Y={(drone_y+self.offset_y):.2f}m Z={(drone_z+self.offset_z):.2f}m)"
            )
        
        # 判断是否需要添加新轨迹点
        add_trajectory_point = False
        
        # 如果轨迹为空，添加第一个点
        if not self.drone_trajectory:
            add_trajectory_point = True
        # 如果与上一个轨迹点坐标不同，且距离方格中心足够近（小于0.1米）
        elif self.drone_trajectory and hasattr(self, 'last_distance_to_center'):
            last_point = self.drone_trajectory[-1]
            # 计算上一个轨迹点对应的网格坐标
            last_grid_col, last_grid_row = self.global_to_grid_coords(last_point[0], last_point[1])
            
            # 只有距离方格中心足够近（小于0.1米）且位置发生变化时才添加新点
            if ((grid_col != last_grid_col or grid_row != last_grid_row) and 
                self.last_distance_to_center < 0.1):
                add_trajectory_point = True
        
        # 添加新的轨迹点
        if add_trajectory_point:
            self.drone_trajectory.append(self.drone_position[:])
        
        # 更新地图显示
        self.map_widget.update()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("无人机喷涂系统地面站")
        # 增加窗口大小
        self.setGeometry(100, 50, 1600, 900)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建状态栏
        self.statusBar().showMessage("就绪")

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧布局（地图和控制面板）
        left_panel = QWidget()
        left_layout = QHBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建地图部件
        self.map_widget = self.create_map_widget()
        left_layout.addWidget(self.map_widget, 7)  # 增大地图比例
        
        # 创建控制面板
        control_panel = self.create_control_panel()
        left_layout.addWidget(control_panel, 2)  # 控制面板占比
        
        # 添加左侧面板到主布局
        main_layout.addWidget(left_panel, 5)  # 左侧占主布局的5/6
        
        # 创建右侧边栏（包含动物检测信息）
        self.right_sidebar = QWidget()
        right_layout = QVBoxLayout(self.right_sidebar)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建动物检测信息显示区域
        wildlife_panel = self.create_wildlife_panel()
        right_layout.addWidget(wildlife_panel)
        
        # 将右侧布局添加到主布局
        main_layout.addWidget(self.right_sidebar, 1)  # 右侧占主布局的1/6
        
        # 创建历史记录面板（初始隐藏）
        self.create_history_panel()
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 状态显示
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("状态: 待机")
        self.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        
        self.waypoint_count_label = QLabel("路径长度: 0 格")
        
        # 添加无人机位置显示
        self.drone_position_label = QLabel("无人机位置: X=0.00m Y=0.00m Z=0.00m")
        self.drone_position_label.setStyleSheet("QLabel { color: #FF8C00; }")  # 橘色
        
        # 添加距离方格中心的距离显示
        self.distance_to_center_label = QLabel("距离方格中心: 0.00m")
        self.distance_to_center_label.setStyleSheet("QLabel { color: #4CAF50; }")  # 绿色
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.waypoint_count_label)
        status_layout.addWidget(self.drone_position_label)
        status_layout.addWidget(self.distance_to_center_label)
        
        layout.addWidget(status_group)
        
        # 禁区设置
        forbidden_group = QGroupBox("禁区设置")
        forbidden_layout = QVBoxLayout(forbidden_group)
        
        forbidden_layout.addWidget(QLabel("点击地图选择3个禁区:"))
        
        self.forbidden_status_label = QLabel("已选择: 0/3")
        forbidden_layout.addWidget(self.forbidden_status_label)
        
        clear_forbidden_btn = QPushButton("清除禁区")
        clear_forbidden_btn.clicked.connect(self.clear_forbidden_zones)
        forbidden_layout.addWidget(clear_forbidden_btn)
        
        layout.addWidget(forbidden_group)
        
        # 路径规划
        planning_group = QGroupBox("路径规划")
        planning_layout = QVBoxLayout(planning_group)
        
        plan_btn = QPushButton("规划混合模式路径")
        plan_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        plan_btn.clicked.connect(self.plan_path)
        planning_layout.addWidget(plan_btn)

        # 添加算法说明
        info_label = QLabel("注：巡查严格相邻，返回沿边缘优先+1.5m优化")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        planning_layout.addWidget(info_label)

        upload_btn = QPushButton("上传航线")
        upload_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        upload_btn.clicked.connect(self.upload_waypoints)
        planning_layout.addWidget(upload_btn)

        # 添加进度条
        self.upload_progress = QProgressBar()
        self.upload_progress.setVisible(False)  # 初始隐藏
        self.upload_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        planning_layout.addWidget(self.upload_progress)
        
        layout.addWidget(planning_group)
        
        # 任务控制
        mission_group = QGroupBox("任务控制")
        mission_layout = QVBoxLayout(mission_group)
        
        self.start_btn = QPushButton("开始")
        self.start_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 15px; }")
        self.start_btn.clicked.connect(self.toggle_mission)
        mission_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        self.reset_btn.clicked.connect(self.reset_mission)
        mission_layout.addWidget(self.reset_btn)
        
        # 添加轨迹控制按钮
        self.clear_trajectory_btn = QPushButton("清除轨迹")
        self.clear_trajectory_btn.setStyleSheet("QPushButton { background-color: #FF8C00; color: white; font-weight: bold; padding: 10px; }")
        self.clear_trajectory_btn.clicked.connect(self.clear_trajectory)
        mission_layout.addWidget(self.clear_trajectory_btn)
        
        layout.addWidget(mission_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
    
    def clear_trajectory(self):
        """清除无人机轨迹"""
        self.drone_trajectory.clear()
        self.map_widget.update()
        QMessageBox.information(self, "提示", "无人机轨迹已清除")
        
    def create_map_widget(self):
        """创建地图显示部件"""
        return MapWidget(self)
        
    def coord_to_position(self, coord_str):
        """将坐标字符串转换为网格位置"""
        # 支持格式：B行 A列 (如 "B1 A9")
        parts = coord_str.strip().split()
        if len(parts) != 2:
            return None

        try:
            # 解析行部分 (B1-B7)
            row_part = parts[0].upper()
            if not row_part.startswith('B'):
                return None
            row_num = int(row_part[1:])
            if not (1 <= row_num <= 7):
                return None
            row = 7 - row_num  # B1对应索引6，B7对应索引0

            # 解析列部分 (A1-A9)
            col_part = parts[1].upper()
            if not col_part.startswith('A'):
                return None
            col_num = int(col_part[1:])
            if not (1 <= col_num <= 9):
                return None
            col = col_num - 1  # A1对应索引0，A9对应索引8

            return (col, row)

        except ValueError:
            return None
        
    def position_to_coord(self, col, row):
        """将网格位置转换为坐标字符串"""
        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            # 行：B1-B7（索引6-0对应B1-B7）
            row_str = f"B{7 - row}"
            # 列：A1-A9（索引0-8对应A1-A9）
            col_str = f"A{col + 1}"
            return f"{row_str} {col_str}"  # 格式：B行 A列
        return None
        
    def grid_to_global_coords(self, col, row, waypoint_index=None, total_waypoints=None):
        """将网格坐标转换为以红点为原点的全局坐标系（单位：米）
        X轴：向前（B1到B7方向）
        Y轴：向左（A9到A1方向）
        每个方格50cm×50cm
        """
        # 计算相对于红点的偏移（以方格为单位）
        # X轴：row方向，B1(row=6)到B7(row=0)，所以X = origin_row - row
        # Y轴：col方向，A9(col=8)到A1(col=0)，所以Y = origin_col - col
        grid_offset_x = self.origin_row - row  # 向前为正
        grid_offset_y = self.origin_col - col  # 向左为正

        # 转换为实际距离（米）
        global_x = grid_offset_x * self.cell_size  # 50cm = 0.5米
        global_y = grid_offset_y * self.cell_size  # 50cm = 0.5米

        # 确定高度：第一个和最后一个航点高度为0，其他为1.22米
        if waypoint_index is not None and total_waypoints is not None:
            if waypoint_index == 0:  # 第一个航点（起飞点）
                # 如果已经记录了无人机初始位置，使用原地起飞，否则使用传统起降点
                if self.coordinate_system_initialized:
                    # 第一个航点是原地起飞，位置与无人机初始位置相同，但高度为巡查高度
                    return -self.offset_x, -self.offset_y, self.survey_height
                else:
                    global_z = self.takeoff_landing_height  # 传统起降高度
            elif waypoint_index == total_waypoints - 1:  # 最后一个航点（降落点）
                global_z = self.takeoff_landing_height  # 起降高度
            else:
                global_z = self.survey_height  # 巡查高度
        else:
            global_z = 0  # 默认高度
        
        # 如果坐标系已初始化，考虑坐标偏移量
        if self.coordinate_system_initialized and waypoint_index is not None:
            # 对于第一个航点，我们已经在前面特殊处理了
            if waypoint_index != 0:
                # 应用坐标系偏移（将原点从红点移动到无人机初始位置）
                global_x = global_x - self.offset_x
                global_y = global_y - self.offset_y
                # Z轴偏移在前面处理高度时已考虑
        
        return global_x, global_y, global_z

    def global_to_grid_coords(self, global_x, global_y):
        """将全局坐标（米）转换回网格坐标"""
        # 如果坐标系已初始化，先应用偏移量还原到红点为原点的坐标系
        if self.coordinate_system_initialized:
            # 还原到红点为原点的坐标系
            adjusted_x = global_x + self.offset_x
            adjusted_y = global_y + self.offset_y
        else:
            adjusted_x = global_x
            adjusted_y = global_y
            
        # 处理特殊情况：原点(0,0)应该精确对应红点位置
        if adjusted_x == 0.0 and adjusted_y == 0.0:
            return self.red_point
            
        # 先转换为方格偏移
        grid_offset_x = adjusted_x / self.cell_size
        grid_offset_y = adjusted_y / self.cell_size

        # 再转换为网格坐标
        col = self.origin_col - grid_offset_y
        row = self.origin_row - grid_offset_x
        
        # 计算精确的网格坐标（包括小数部分）
        exact_col = col
        exact_row = row
        
        # 计算到最近网格中心的距离
        col_center = round(col)
        row_center = round(row)
        
        # 计算中心点对应的全局坐标
        center_global_x = (self.origin_row - row_center) * self.cell_size
        center_global_y = (self.origin_col - col_center) * self.cell_size
        
        if self.coordinate_system_initialized:
            # 应用坐标偏移
            center_global_x = center_global_x - self.offset_x
            center_global_y = center_global_y - self.offset_y
        
        # 计算距离
        distance_to_center = ((global_x - center_global_x) ** 2 + (global_y - center_global_y) ** 2) ** 0.5
        
        # 保存距离和准确坐标供使用
        self.last_distance_to_center = distance_to_center
        self.exact_grid_coords = (exact_col, exact_row)
        
        # 返回最近的网格坐标（四舍五入）
        return int(round(col)), int(round(row))

    def clear_forbidden_zones(self):
        """清除所有禁区"""
        self.forbidden_zones.clear()
        self.update_forbidden_status()
        self.map_widget.update()
        
    def update_forbidden_status(self):
        """更新禁区状态显示"""
        count = len(self.forbidden_zones)
        self.forbidden_status_label.setText(f"已选择: {count}/3")
        
    def add_forbidden_zone(self, col, row):
        """添加禁区"""
        if len(self.forbidden_zones) >= 3:
            QMessageBox.warning(self, "警告", "最多只能设置3个禁区")
            return False
        
        # 检查是否已经是禁区
        if (col, row) in self.forbidden_zones:
            return False
        
        # 检查是否是红点位置
        if (col, row) == self.red_point:
            QMessageBox.warning(self, "警告", "不能在起降点设置禁区")
            return False
        
        # 添加禁区
        self.forbidden_zones.append((col, row))
        self.update_forbidden_status()
        return True
        
    def find_detour_to_landing_point_with_repeats(self, start, target):
        """寻找从起点到目标点的绕行路径，允许重复航点"""
        # 首先尝试原有的绕行策略
        detour_path = self.find_detour_to_landing_point(start, target)
        if detour_path:
            return detour_path
        
        # 如果原有策略失败，允许通过已访问的航点进行绕行
        for waypoint in self.waypoints:
            if waypoint != start and waypoint not in self.forbidden_zones:
                # 检查从起点到已访问航点，再到目标点的路径是否安全
                if (not self.path_crosses_forbidden_zone(start, waypoint) and
                    not self.path_crosses_forbidden_zone(waypoint, target)):
                    return [waypoint, target]
        
        # 如果仍然失败，尝试通过多个已访问航点的组合
        for i, waypoint1 in enumerate(self.waypoints):
            if waypoint1 != start and waypoint1 not in self.forbidden_zones:
                for j, waypoint2 in enumerate(self.waypoints):
                    if (waypoint2 != start and waypoint2 != waypoint1 and 
                        waypoint2 not in self.forbidden_zones):
                        # 检查三段路径是否都安全
                        if (not self.path_crosses_forbidden_zone(start, waypoint1) and
                            not self.path_crosses_forbidden_zone(waypoint1, waypoint2) and
                            not self.path_crosses_forbidden_zone(waypoint2, target)):
                            return [waypoint1, waypoint2, target]
        
        return None
            
        if (col, row) == self.red_point:
            QMessageBox.warning(self, "警告", "不能将红点设为禁区")
            return False
            
        if (col, row) in self.forbidden_zones:
            QMessageBox.warning(self, "警告", "该位置已是禁区")
            return False
            
        self.forbidden_zones.append((col, row))
        self.update_forbidden_status()
        self.map_widget.update()
        return True
        
    def plan_path(self):
        """规划路径 - 巡查严格相邻移动，返回可灵活移动但避开禁区顶点"""
        if len(self.forbidden_zones) < 3:
            QMessageBox.warning(self, "警告", "请先选择3个禁区！")
            return

        # 清空之前的航点和返回路径索引
        self.waypoints = []
        self.return_path_start_index = -1

        # 生成所有需要访问的点（除了禁区）
        all_points = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                point = (col, row)
                if point not in self.forbidden_zones:
                    all_points.append(point)

        # 确保起降点在可访问点列表中
        if self.red_point not in all_points:
            QMessageBox.warning(self, "警告", "起降点位置被禁区阻挡，无法规划路径！")
            return

        # 从起降点开始
        self.waypoints.append(self.red_point)

        # 从待访问点中移除起始点
        remaining_points = [p for p in all_points if p != self.red_point]

        # 第一阶段：使用严格相邻移动进行巡查
        survey_completed_points = self.plan_adjacent_survey(remaining_points.copy())

        # 计算巡查阶段未能访问的点
        visited_in_survey = set(self.waypoints)
        unvisited_points = [p for p in all_points if p not in visited_in_survey]

        # 第二阶段：返回路径中遍历剩余点（允许灵活移动）
        if not self.add_flexible_return_path(unvisited_points):
            QMessageBox.warning(self, "警告", "无法规划返回路径！")
            return

        # 更新显示
        total_distance = self.calculate_total_distance()
        self.waypoint_count_label.setText(f"路径长度: {len(self.waypoints)} 格，总距离: {total_distance:.1f} 米")
        self.map_widget.update()

        # 验证是否遍历了所有非禁区点
        visited_points = set(self.waypoints)
        expected_points = set(all_points)
        is_closed_loop = (len(self.waypoints) >= 2 and
                         self.waypoints[0] == self.waypoints[-1] == self.red_point)

        if visited_points == expected_points and is_closed_loop:
            survey_points = len([p for p in self.waypoints[:self.return_path_start_index+1] if p in all_points])
            return_points = len(self.waypoints) - self.return_path_start_index - 1
            used_fallback = self.check_fallback_usage()

            fallback_info = "（含相邻容错）" if used_fallback else "（灵活移动）"

            QMessageBox.information(self, "成功",
                f"混合模式路径规划完成！\n"
                f"• 总航点：{len(self.waypoints)}个\n"
                f"• 巡查航点：{survey_points}个（严格相邻）\n"
                f"• 返回航点：{return_points}个{fallback_info}\n"
                f"• 遍历方块：{len(all_points)}个（50cm×50cm）\n"
                f"• 总距离：{total_distance:.1f}米\n"
                f"• 已形成闭环")
        else:
            missing_points = expected_points - visited_points
            if missing_points:
                QMessageBox.warning(self, "警告", f"路径规划不完整，遗漏了{len(missing_points)}个方块！")
            elif not is_closed_loop:
                QMessageBox.warning(self, "警告", "路径未形成闭环！")
        
    def find_optimal_start_point(self, all_points):
        """寻找最优起始点，强制从B1 A9位置开始（与降落点相同）"""
        # 强制起始点为B1 A9位置，与降落点相同 (A9列B1行，对应坐标(8, 6))
        start_point = self.red_point  # 使用与降落点相同的位置 B1 A9

        # 检查B1 A9位置是否在可访问点列表中
        if start_point in all_points:
            return start_point
        else:
            # 如果B1 A9位置不可访问（比如是禁区），返回None
            return None

    def plan_adjacent_survey(self, remaining_points):
        """第一阶段：使用严格相邻移动进行巡查"""
        unvisited = set(remaining_points)

        while unvisited:
            current_pos = self.waypoints[-1]

            # 寻找相邻的未访问点
            adjacent_point = self.find_adjacent_unvisited(current_pos, unvisited)

            if adjacent_point:
                # 直接移动到相邻点
                self.waypoints.append(adjacent_point)
                unvisited.remove(adjacent_point)
            else:
                # 寻找通过相邻移动能到达的最近未访问点
                path_to_unvisited = self.find_adjacent_path_to_unvisited(current_pos, unvisited)
                if path_to_unvisited:
                    # 添加路径上的所有点
                    for point in path_to_unvisited:
                        self.waypoints.append(point)
                        if point in unvisited:
                            unvisited.remove(point)
                else:
                    # 无法通过相邻移动到达任何未访问点，结束巡查阶段
                    break

        return len(unvisited) == 0

    def find_adjacent_unvisited(self, current_pos, unvisited):
        """寻找相邻的未访问点，优先选择与当前移动方向一致的点以减少拐弯"""
        # 获取当前移动方向
        current_direction = self.get_current_direction()
        
        # 定义方向优先级：优先选择与当前方向一致的方向，然后是直角方向，最后是反方向
        direction_priorities = self.get_direction_priorities(current_direction)
        
        # 按优先级顺序检查相邻点
        for dx, dy in direction_priorities:
            next_col = current_pos[0] + dx
            next_row = current_pos[1] + dy
            next_point = (next_col, next_row)

            if (0 <= next_col < self.grid_cols and
                0 <= next_row < self.grid_rows and
                next_point in unvisited and
                next_point not in self.forbidden_zones):
                return next_point

        return None

    def get_current_direction(self):
        """获取当前移动方向，用于优化路径规划减少拐弯"""
        if len(self.waypoints) < 2:
            return None  # 没有足够的航点来确定方向
        
        # 计算最后两个航点之间的方向向量
        last_point = self.waypoints[-1]
        second_last_point = self.waypoints[-2]
        
        dx = last_point[0] - second_last_point[0]
        dy = last_point[1] - second_last_point[1]
        
        return (dx, dy)
    
    def get_direction_priorities(self, current_direction):
        """根据当前移动方向获取方向优先级列表，减少连续拐弯"""
        # 定义所有可能的移动方向：上、右、下、左
        all_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        if current_direction is None:
            # 如果没有当前方向，使用默认顺序（优先向右和向上，便于形成规律路径）
            return [(1, 0), (0, 1), (0, -1), (-1, 0)]  # 右、上、下、左
        
        # 将当前方向放在最前面（继续直行）
        priorities = [current_direction]
        
        # 添加垂直方向（90度转弯）
        perpendicular_directions = []
        if current_direction == (0, 1):  # 当前向上
            perpendicular_directions = [(1, 0), (-1, 0)]  # 右、左
        elif current_direction == (1, 0):  # 当前向右
            perpendicular_directions = [(0, 1), (0, -1)]  # 上、下
        elif current_direction == (0, -1):  # 当前向下
            perpendicular_directions = [(1, 0), (-1, 0)]  # 右、左
        elif current_direction == (-1, 0):  # 当前向左
            perpendicular_directions = [(0, 1), (0, -1)]  # 上、下
        
        priorities.extend(perpendicular_directions)
        
        # 最后添加反方向（180度转弯，最不优先）
        opposite_direction = (-current_direction[0], -current_direction[1])
        priorities.append(opposite_direction)
        
        # 确保所有方向都包含在内，去除重复
        final_priorities = []
        for direction in priorities:
            if direction in all_directions and direction not in final_priorities:
                final_priorities.append(direction)
        
        # 添加任何遗漏的方向
        for direction in all_directions:
            if direction not in final_priorities:
                final_priorities.append(direction)
        
        return final_priorities

    def find_adjacent_path_to_unvisited(self, start, unvisited):
        """使用优化的BFS寻找通过相邻移动到达未访问点的路径，优先选择拐弯次数少的路径"""
        if not unvisited:
            return None
        
        # 为每个未访问点计算路径，并按拐弯次数和距离排序
        target_paths = []
        
        for target in unvisited:
            path = self.find_adjacent_path(start, target)
            if path:
                # 计算路径的拐弯次数
                turns = self.count_turns_in_path([start] + path)
                # 计算曼哈顿距离作为次要排序条件
                distance = abs(target[0] - start[0]) + abs(target[1] - start[1])
                target_paths.append((turns, distance, len(path), target, path))
        
        if not target_paths:
            return None
        
        # 按拐弯次数、距离、路径长度排序，优先选择拐弯少的路径
        target_paths.sort(key=lambda x: (x[0], x[1], x[2]))
        
        # 返回最优路径
        return target_paths[0][4]  # 返回path

    def find_adjacent_path(self, start, end):
        """使用优化的BFS寻找两点间的相邻移动路径，优先选择拐弯次数较少的路径"""
        import heapq

        if start == end:
            return [end]

        # 使用优先队列，优先级为：(拐弯次数, 路径长度, 当前位置, 路径, 上一个方向)
        queue = [(0, 0, start, [start], None)]
        visited = {start: (0, 0)}  # 位置 -> (最少拐弯次数, 最短路径长度)

        while queue:
            turns, length, current, path, last_direction = heapq.heappop(queue)

            # 如果到达目标点
            if current == end:
                return path[1:]  # 不包括起始点

            # 获取当前移动方向的优先级
            current_direction = self.get_current_direction() if len(self.waypoints) >= 2 else None
            if last_direction is not None:
                current_direction = last_direction
            
            direction_priorities = self.get_direction_priorities(current_direction)

            # 检查所有相邻方向，按优先级顺序
            for dx, dy in direction_priorities:
                next_col = current[0] + dx
                next_row = current[1] + dy
                next_point = (next_col, next_row)

                if (0 <= next_col < self.grid_cols and
                    0 <= next_row < self.grid_rows and
                    next_point not in self.forbidden_zones):

                    # 计算新的拐弯次数
                    new_turns = turns
                    if last_direction is not None and (dx, dy) != last_direction:
                        new_turns += 1
                    
                    new_length = length + 1
                    new_path = path + [next_point]

                    # 检查是否应该访问这个点
                    should_visit = True
                    if next_point in visited:
                        prev_turns, prev_length = visited[next_point]
                        # 只有在拐弯次数更少，或拐弯次数相同但路径更短时才访问
                        if new_turns > prev_turns or (new_turns == prev_turns and new_length >= prev_length):
                            should_visit = False

                    if should_visit:
                        visited[next_point] = (new_turns, new_length)
                        heapq.heappush(queue, (new_turns, new_length, next_point, new_path, (dx, dy)))

        return None

    def add_flexible_return_path(self, unvisited_points):
        """第二阶段：返回路径中遍历剩余点（优先沿边缘返航）"""
        if not self.waypoints:
            return False

        # 记录返回路径的起始索引
        self.return_path_start_index = len(self.waypoints) - 1

        current_pos = self.waypoints[-1]
        remaining_unvisited = unvisited_points.copy()

        if not remaining_unvisited:
            # 如果没有剩余点，直接沿边缘返回起降点
            return self.add_edge_return_to_landing(current_pos)

        # 优先沿边缘遍历剩余点并返回起降点
        edge_return_path = self.find_edge_return_path(current_pos, remaining_unvisited)

        if edge_return_path:
            # 添加沿边缘的返回路径
            for point in edge_return_path:
                self.waypoints.append(point)
            return True
        else:
            # 如果沿边缘路径失败，使用灵活移动作为备选
            print("警告：无法沿边缘返航，使用灵活移动")
            return self.add_flexible_fallback_return(current_pos, remaining_unvisited)

    def find_edge_return_path(self, start_pos, unvisited_points):
        """寻找沿边缘遍历剩余点并返回起降点的路径"""
        path = []
        current = start_pos
        remaining = unvisited_points.copy()

        # 首先移动到最近的边缘
        edge_entry_path = self.move_to_nearest_edge(current)
        if edge_entry_path:
            path.extend(edge_entry_path)
            current = edge_entry_path[-1]
            # 移除路径中访问的未访问点
            for point in edge_entry_path:
                if point in remaining:
                    remaining.remove(point)

        # 沿边缘遍历剩余的未访问点
        while remaining:
            # 检查1.5米优化
            if self.is_close_and_safe_to_landing(current):
                print(f"优化：沿边缘返航时，剩余{len(remaining)}个点，但已接近起降点，直接返航")
                break

            # 寻找沿边缘到达的下一个未访问点
            next_edge_point = self.find_next_edge_unvisited(current, remaining)
            if next_edge_point:
                edge_segment = self.find_edge_path(current, next_edge_point)
                if edge_segment:
                    path.extend(edge_segment)
                    current = edge_segment[-1]
                    # 移除路径中访问的未访问点
                    for point in edge_segment:
                        if point in remaining:
                            remaining.remove(point)
                else:
                    # 无法沿边缘到达，移除该点
                    remaining.remove(next_edge_point)
            else:
                # 没有更多可沿边缘到达的点
                break

        # 沿边缘返回起降点
        edge_return_segment = self.find_edge_path_to_landing(current)
        if edge_return_segment:
            path.extend(edge_return_segment)

        return path if path else None

    def move_to_nearest_edge(self, start_pos):
        """移动到最近的边缘"""
        # 如果已经在边缘，直接返回
        if self.is_on_edge(start_pos):
            return []

        # 寻找最近的边缘点
        nearest_edge_point = self.find_nearest_edge_point(start_pos)
        if nearest_edge_point:
            return self.get_shortest_path_between(start_pos, nearest_edge_point)

        return None

    def is_on_edge(self, pos):
        """检查位置是否在网格边缘"""
        col, row = pos
        return (col == 0 or col == self.grid_cols - 1 or
                row == 0 or row == self.grid_rows - 1)

    def find_nearest_edge_point(self, start_pos):
        """寻找最近的边缘点"""
        col, row = start_pos
        edge_points = []

        # 添加四个边缘的候选点
        # 上边缘
        if row > 0:
            edge_points.append((col, 0))
        # 下边缘
        if row < self.grid_rows - 1:
            edge_points.append((col, self.grid_rows - 1))
        # 左边缘
        if col > 0:
            edge_points.append((0, row))
        # 右边缘
        if col < self.grid_cols - 1:
            edge_points.append((self.grid_cols - 1, row))

        # 过滤掉禁区点，找到最近的可达边缘点
        valid_edge_points = [p for p in edge_points if p not in self.forbidden_zones]

        if not valid_edge_points:
            return None

        # 返回距离最近的边缘点
        return min(valid_edge_points,
                  key=lambda p: abs(p[0] - col) + abs(p[1] - row))

    def find_next_edge_unvisited(self, current, remaining):
        """寻找沿边缘可到达的下一个未访问点"""
        if not remaining:
            return None

        # 按距离排序未访问点
        sorted_remaining = sorted(remaining,
                                 key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]))

        for target in sorted_remaining:
            # 检查是否可以沿边缘到达
            if self.can_reach_via_edge(current, target):
                return target

        return None

    def can_reach_via_edge(self, start, target):
        """检查是否可以沿边缘到达目标点"""
        # 简单实现：检查是否可以通过边缘路径到达
        edge_path = self.find_edge_path(start, target)
        return edge_path is not None

    def find_edge_path(self, start, end):
        """寻找沿边缘的路径"""
        # 如果起点或终点不在边缘，先移动到边缘
        path = []
        current = start

        # 确保起点在边缘
        if not self.is_on_edge(current):
            to_edge = self.move_to_nearest_edge(current)
            if to_edge:
                path.extend(to_edge)
                current = to_edge[-1]
            else:
                return None

        # 沿边缘移动到目标
        edge_segment = self.find_edge_path_direct(current, end)
        if edge_segment:
            path.extend(edge_segment)

        return path if path else None

    def find_edge_path_direct(self, start, end):
        """直接沿边缘寻找路径"""
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == end:
                return path[1:]  # 不包括起始点

            # 检查相邻的边缘点
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_col = current[0] + dx
                next_row = current[1] + dy
                next_point = (next_col, next_row)

                if (0 <= next_col < self.grid_cols and
                    0 <= next_row < self.grid_rows and
                    next_point not in visited and
                    next_point not in self.forbidden_zones and
                    self.is_on_edge(next_point)):  # 必须在边缘

                    visited.add(next_point)
                    queue.append((next_point, path + [next_point]))

        return None

    def find_edge_path_to_landing(self, start):
        """沿边缘返回起降点"""
        if start == self.red_point:
            return [self.red_point]

        # 检查1.5米优化
        if self.is_close_and_safe_to_landing(start):
            return [self.red_point]

        # 沿边缘寻找到起降点的路径
        edge_path = self.find_edge_path_direct(start, self.red_point)
        if edge_path:
            return edge_path + [self.red_point]

        # 如果沿边缘无法直接到达，先到最近的边缘点再到起降点
        nearest_to_landing = self.find_nearest_edge_to_landing()
        if nearest_to_landing and nearest_to_landing != start:
            to_nearest = self.find_edge_path_direct(start, nearest_to_landing)
            from_nearest = self.get_shortest_path_between(nearest_to_landing, self.red_point)
            if to_nearest and from_nearest:
                return to_nearest + from_nearest

        return None

    def find_nearest_edge_to_landing(self):
        """寻找距离起降点最近的边缘点"""
        edge_points = []

        # 收集所有边缘点
        for col in range(self.grid_cols):
            for row in range(self.grid_rows):
                if self.is_on_edge((col, row)) and (col, row) not in self.forbidden_zones:
                    edge_points.append((col, row))

        if not edge_points:
            return None

        # 返回距离起降点最近的边缘点
        return min(edge_points,
                  key=lambda p: abs(p[0] - self.red_point[0]) + abs(p[1] - self.red_point[1]))

    def add_edge_return_to_landing(self, current_pos):
        """沿边缘返回起降点"""
        if current_pos == self.red_point:
            return True

        edge_return_path = self.find_edge_path_to_landing(current_pos)
        if edge_return_path:
            for point in edge_return_path:
                self.waypoints.append(point)
            return True

        # 如果沿边缘失败，使用直接路径
        print("警告：无法沿边缘返回起降点，使用直接路径")
        return self.add_shortest_return_to_landing(current_pos)

    def add_flexible_fallback_return(self, start_pos, unvisited_points):
        """灵活移动的备选返回方案"""
        current_pos = start_pos
        remaining_unvisited = unvisited_points.copy()

        while remaining_unvisited:
            if self.is_close_and_safe_to_landing(current_pos):
                break

            nearest_point = min(remaining_unvisited,
                               key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))

            path_to_point = self.get_shortest_path_between(current_pos, nearest_point)
            if path_to_point:
                for point in path_to_point:
                    self.waypoints.append(point)
                    if point in remaining_unvisited:
                        remaining_unvisited.remove(point)
                current_pos = path_to_point[-1]
            else:
                remaining_unvisited.remove(nearest_point)

        return self.add_shortest_return_to_landing(current_pos)

    def find_shortest_return_path(self, start_pos, unvisited_points):
        """寻找遍历所有剩余点并返回起降点的最短路径"""
        if not unvisited_points:
            return self.get_shortest_path_to_landing(start_pos)

        # 对于少量剩余点，使用动态规划寻找最短路径
        if len(unvisited_points) <= 6:  # 限制计算复杂度
            return self.solve_tsp_return(start_pos, unvisited_points)
        else:
            # 对于较多剩余点，使用优化的贪心算法
            return self.solve_greedy_shortest_return(start_pos, unvisited_points)

    def solve_tsp_return(self, start_pos, unvisited_points):
        """使用动态规划解决小规模TSP返回问题"""
        from itertools import permutations

        best_path = None
        min_distance = float('inf')

        # 尝试所有可能的访问顺序
        for perm in permutations(unvisited_points):
            path = []
            current = start_pos
            total_distance = 0
            valid_path = True

            # 计算访问所有点的路径
            for target in perm:
                segment = self.get_shortest_path_between(current, target)
                if segment is None:
                    valid_path = False
                    break

                path.extend(segment)
                total_distance += len(segment)
                current = target

            if not valid_path:
                continue

            # 添加返回起降点的路径
            final_segment = self.get_shortest_path_to_landing(current)
            if final_segment is None:
                continue

            path.extend(final_segment)
            total_distance += len(final_segment)

            # 检查是否是最短路径
            if total_distance < min_distance:
                min_distance = total_distance
                best_path = path

        return best_path

    def solve_greedy_shortest_return(self, start_pos, unvisited_points):
        """使用优化贪心算法寻找较短的返回路径"""
        path = []
        current = start_pos
        remaining = unvisited_points.copy()

        while remaining:
            # 检查1.5米优化
            if self.is_close_and_safe_to_landing(current):
                print(f"优化：剩余{len(remaining)}个点，但已接近起降点，直接返航")
                break

            # 寻找到最近点的最短路径
            best_next = None
            best_path = None
            min_cost = float('inf')

            for target in remaining:
                segment = self.get_shortest_path_between(current, target)
                if segment:
                    # 计算总成本：到达成本 + 从该点到起降点的估计成本
                    reach_cost = len(segment)
                    return_cost = abs(target[0] - self.red_point[0]) + abs(target[1] - self.red_point[1])
                    total_cost = reach_cost + return_cost * 0.5  # 给返回成本较小权重

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_next = target
                        best_path = segment

            if best_next and best_path:
                path.extend(best_path)
                remaining.remove(best_next)
                current = best_next
            else:
                # 无法到达任何剩余点
                break

        # 添加返回起降点的路径
        final_segment = self.get_shortest_path_to_landing(current)
        if final_segment:
            path.extend(final_segment)

        return path if path else None

    def add_greedy_return_path(self, start_pos, unvisited_points):
        """备选的贪心返回路径算法"""
        current_pos = start_pos
        remaining_unvisited = unvisited_points.copy()

        while remaining_unvisited:
            if self.is_close_and_safe_to_landing(current_pos):
                break

            nearest_point = min(remaining_unvisited,
                               key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))

            path_to_point = self.find_flexible_path(current_pos, nearest_point)
            if not path_to_point:
                path_to_point = self.find_adjacent_path(current_pos, nearest_point)

            if path_to_point:
                for point in path_to_point:
                    self.waypoints.append(point)
                    if point in remaining_unvisited:
                        remaining_unvisited.remove(point)
                current_pos = path_to_point[-1]
            else:
                remaining_unvisited.remove(nearest_point)

        return self.add_shortest_return_to_landing(current_pos)

    def get_shortest_path_between(self, start, end):
        """获取两点间的最短路径"""
        path = self.find_flexible_path(start, end)
        if path:
            return path
        return self.find_adjacent_path(start, end)

    def get_shortest_path_to_landing(self, start):
        """获取到起降点的最短路径"""
        if start == self.red_point:
            return [self.red_point]

        if self.is_close_and_safe_to_landing(start):
            return [self.red_point]

        path = self.find_flexible_path(start, self.red_point)
        if path:
            return path

        return self.find_adjacent_path(start, self.red_point)

    def add_shortest_return_to_landing(self, current_pos):
        """添加到起降点的最短路径"""
        if current_pos == self.red_point:
            return True

        return_path = self.get_shortest_path_to_landing(current_pos)
        if return_path:
            for point in return_path:
                self.waypoints.append(point)
            return True

        return False

    def calculate_total_distance(self):
        """计算总路程（实际米数）"""
        if len(self.waypoints) < 2:
            return 0.0

        total_distance = 0
        for i in range(len(self.waypoints) - 1):
            p1 = self.waypoints[i]
            p2 = self.waypoints[i + 1]
            grid_distance = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
            total_distance += grid_distance

        # 转换为实际米数
        return total_distance * self.cell_size

    def is_close_and_safe_to_landing(self, current_pos):
        """检查是否距离起降点1.5米内且直线路径安全"""
        # 计算到起降点的实际距离（每个方块50cm = 0.5米）
        grid_distance = abs(current_pos[0] - self.red_point[0]) + abs(current_pos[1] - self.red_point[1])
        distance_meters = grid_distance * self.cell_size  # 转换为实际米数

        # 检查是否在1.5米范围内
        if distance_meters <= 1.5:
            # 检查直线路径是否安全（不经过禁飞区）
            if not self.path_crosses_forbidden_zone(current_pos, self.red_point):
                print(f"优化：距离起降点{distance_meters:.1f}米，直线安全，直接返航")
                return True

        return False

    def check_fallback_usage(self):
        """检查是否使用了相邻移动容错机制"""
        # 这里可以通过检查路径特征来判断是否使用了容错
        # 简单实现：检查返回路径中是否有只能通过相邻移动到达的路径段
        if self.return_path_start_index < 0 or len(self.waypoints) <= self.return_path_start_index + 1:
            return False

        # 检查返回路径中的移动是否都是相邻的
        return_waypoints = self.waypoints[self.return_path_start_index + 1:]
        if len(return_waypoints) < 2:
            return False

        # 如果返回路径中有非相邻移动，说明使用了灵活移动
        # 如果全部都是相邻移动，可能使用了容错机制
        all_adjacent = True
        for i in range(len(return_waypoints) - 1):
            p1 = return_waypoints[i]
            p2 = return_waypoints[i + 1]
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])

            # 如果不是相邻移动（对角线或更远距离）
            if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
                all_adjacent = False
                break

        # 如果返回路径较长且全部是相邻移动，可能使用了容错
        return all_adjacent and len(return_waypoints) > 3

    def find_flexible_path(self, start, end):
        """寻找灵活移动路径（允许对角线等），但不能经过禁区顶点"""
        # 检查是否可以直接到达
        if not self.path_crosses_forbidden_zone(start, end):
            return [end]

        # 使用A*算法寻找避开禁区的路径
        from collections import deque
        import heapq

        open_set = [(0, start, [])]
        closed_set = set()

        while open_set:
            f_score, current, path = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == end:
                return path + [end]

            # 检查8个方向的移动（包括对角线）
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),    # 上右下左
                (1, 1), (1, -1), (-1, 1), (-1, -1)   # 对角线
            ]

            for dx, dy in directions:
                next_col = current[0] + dx
                next_row = current[1] + dy
                next_point = (next_col, next_row)

                if (0 <= next_col < self.grid_cols and
                    0 <= next_row < self.grid_rows and
                    next_point not in closed_set and
                    next_point not in self.forbidden_zones and
                    not self.path_crosses_forbidden_zone(current, next_point)):

                    g_score = len(path) + 1
                    h_score = abs(next_point[0] - end[0]) + abs(next_point[1] - end[1])
                    f_score = g_score + h_score

                    new_path = path + [next_point] if next_point != end else path
                    heapq.heappush(open_set, (f_score, next_point, new_path))

        return None

    def plan_complete_coverage(self, remaining_points):
        """完全覆盖规划，确保访问所有剩余点，只允许90度转弯"""
        unvisited = set(remaining_points)
        
        while unvisited:
            # 从当前位置找到最佳的下一个点（只允许90度转弯）
            current_pos = self.waypoints[-1] if self.waypoints else self.red_point
            current_col, current_row = current_pos
            
            # 获取当前移动方向（如果有的话）
            current_direction = None
            if len(self.waypoints) >= 2:
                prev_pos = self.waypoints[-2]
                prev_col, prev_row = prev_pos
                
                if prev_col == current_col:  # 垂直移动
                    current_direction = 'vertical'
                elif prev_row == current_row:  # 水平移动
                    current_direction = 'horizontal'
            
            # 寻找可行的下一个点
            best_next = None
            best_score = float('inf')  # 较小的分数更好
            
            # 检查四个方向的相邻点
            directions = [
                (0, 1),   # 上
                (1, 0),   # 右
                (0, -1),  # 下
                (-1, 0)   # 左
            ]
            
            for dx, dy in directions:
                next_col = current_col + dx
                next_row = current_row + dy
                next_point = (next_col, next_row)
                
                # 检查点是否有效且未访问
                if (0 <= next_col < self.grid_cols and 
                    0 <= next_row < self.grid_rows and
                    next_point in unvisited and
                    next_point not in self.forbidden_zones and
                    next_point != self.red_point):
                    
                    # 检查路径是否安全
                    if not self.path_crosses_forbidden_zone(current_pos, next_point):
                        # 计算分数（优先考虑保持当前方向，其次考虑未访问点数量）
                        score = 0
                        
                        # 检查是否需要转弯
                        next_direction = 'horizontal' if dy == 0 else 'vertical'
                        if current_direction and current_direction != next_direction:
                            score += 100  # 转弯惩罚
                        
                        # 计算该方向上连续可访问的点数量（越多越好）
                        continuous_points = 0
                        check_col, check_row = next_col, next_row
                        while True:
                            check_col += dx
                            check_row += dy
                            check_point = (check_col, check_row)
                            
                            if (0 <= check_col < self.grid_cols and 
                                0 <= check_row < self.grid_rows and
                                check_point in unvisited and
                                check_point not in self.forbidden_zones and
                                check_point != self.red_point and
                                not self.path_crosses_forbidden_zone(next_point, check_point)):
                                continuous_points += 1
                            else:
                                break
                        
                        # 分数计算：转弯惩罚 - 连续点奖励
                        score -= continuous_points * 10
                        
                        if score < best_score:
                            best_score = score
                            best_next = next_point
            
            if best_next:
                self.waypoints.append(best_next)
                unvisited.remove(best_next)
            else:
                # 如果没有直接可达的点，尝试通过中间点到达
                if not self.find_path_through_intermediate(unvisited):
                    # 如果仍然无法安全到达任何剩余点，停止规划
                    break
                    
    def find_path_through_intermediate(self, unvisited):
        """通过中间点寻找到达未访问点的路径，确保完全避开禁飞区"""
        current_pos = self.waypoints[-1] if self.waypoints else self.red_point
        
        # 按距离排序未访问点，优先处理较近的点
        sorted_targets = sorted(unvisited, key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
        
        for target in sorted_targets:
            # 尝试找到安全的中间点路径
            best_path = self.find_safe_intermediate_path(current_pos, target, unvisited)
            if best_path:
                # 添加找到的安全路径
                for point in best_path:
                    if point in unvisited:
                        self.waypoints.append(point)
                        unvisited.discard(point)
                    elif point != current_pos:  # 避免重复添加当前位置
                        self.waypoints.append(point)
                return True
        return False
        
    def find_safe_intermediate_path(self, start, target, unvisited):
        """寻找从起点到目标点的安全中间路径"""
        # 尝试不同的中间点策略
        strategies = [
            self.try_direct_intermediate,
            self.try_corner_intermediate,
            self.try_edge_intermediate
        ]
        
        for strategy in strategies:
            path = strategy(start, target, unvisited)
            if path:
                return path
        return None
        
    def try_direct_intermediate(self, start, target, unvisited):
        """尝试直接中间点路径"""
        # 尝试所有可能的中间点
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                intermediate = (col, row)
                if (intermediate not in self.forbidden_zones and 
                    intermediate != self.red_point and
                    intermediate not in self.waypoints):
                    
                    # 检查从起点到中间点，再到目标点的路径是否安全
                    if (not self.path_crosses_forbidden_zone(start, intermediate) and
                        not self.path_crosses_forbidden_zone(intermediate, target)):
                        
                        if intermediate in unvisited:
                            return [intermediate, target]
                        else:
                            return [intermediate, target]
        return None
        
    def try_corner_intermediate(self, start, target, unvisited):
        """尝试通过角落点的路径"""
        start_col, start_row = start
        target_col, target_row = target
        
        # 尝试两个角落点：(start_col, target_row) 和 (target_col, start_row)
        corners = [(start_col, target_row), (target_col, start_row)]
        
        for corner in corners:
            if (corner not in self.forbidden_zones and 
                corner != self.red_point and
                corner not in self.waypoints):
                
                # 检查通过角落点的路径是否安全
                if (not self.path_crosses_forbidden_zone(start, corner) and
                    not self.path_crosses_forbidden_zone(corner, target)):
                    
                    if corner in unvisited:
                        return [corner, target]
                    else:
                        return [corner, target]
        return None
        
    def try_edge_intermediate(self, start, target, unvisited):
        """尝试通过边缘点的路径"""
        # 尝试沿着网格边缘寻找安全路径
        edge_points = []
        
        # 添加边缘点
        for col in range(self.grid_cols):
            edge_points.extend([(col, 0), (col, self.grid_rows-1)])
        for row in range(self.grid_rows):
            edge_points.extend([(0, row), (self.grid_cols-1, row)])
        
        # 移除重复点和禁区点
        edge_points = list(set(edge_points))
        edge_points = [p for p in edge_points if p not in self.forbidden_zones and p != self.red_point]
        
        for edge_point in edge_points:
            if edge_point not in self.waypoints:
                # 检查通过边缘点的路径是否安全
                if (not self.path_crosses_forbidden_zone(start, edge_point) and
                    not self.path_crosses_forbidden_zone(edge_point, target)):
                    
                    if edge_point in unvisited:
                        return [edge_point, target]
                    else:
                        return [edge_point, target]
        return None
                    
    def is_safe_path_to_point(self, target_point):
        """检查到目标点的路径是否安全（不经过禁区）"""
        if not self.waypoints:
            return True
            
        last_point = self.waypoints[-1]
        return not self.path_crosses_forbidden_zone(last_point, target_point)
        
    def path_crosses_forbidden_zone(self, start, end):
        """检查路径是否穿过禁区（包括禁区的边缘和顶点）"""
        start_col, start_row = start
        end_col, end_row = end

        # 如果起点或终点在禁区内，直接返回True
        if start in self.forbidden_zones or end in self.forbidden_zones:
            return True

        # 直接检查路径线段是否与任何禁区方块相交
        for forbidden_col, forbidden_row in self.forbidden_zones:
            # 检查路径是否与禁区方块相交
            if self.line_intersects_square(start_col, start_row, end_col, end_row,
                                         forbidden_col, forbidden_row):
                return True

        return False

    def line_intersects_square(self, x1, y1, x2, y2, square_col, square_row):
        """检查线段是否与方块相交（包括边缘和顶点）"""
        # 方块的边界（每个方块占据从(col, row)到(col+1, row+1)的区域）
        square_left = square_col
        square_right = square_col + 1
        square_top = square_row
        square_bottom = square_row + 1

        # 使用线段与矩形相交的算法
        # 检查线段是否与矩形的任何边相交

        # 如果线段的两个端点都在矩形的同一侧，则不相交
        if ((x1 < square_left and x2 < square_left) or
            (x1 > square_right and x2 > square_right) or
            (y1 < square_top and y2 < square_top) or
            (y1 > square_bottom and y2 > square_bottom)):
            return False

        # 如果线段的任一端点在矩形内，则相交
        if (square_left <= x1 <= square_right and square_top <= y1 <= square_bottom):
            return True
        if (square_left <= x2 <= square_right and square_top <= y2 <= square_bottom):
            return True

        # 检查线段是否与矩形的边相交
        # 使用参数方程检查相交
        dx = x2 - x1
        dy = y2 - y1

        if dx != 0:
            # 检查与左边和右边的相交
            t_left = (square_left - x1) / dx
            t_right = (square_right - x1) / dx

            for t in [t_left, t_right]:
                if 0 <= t <= 1:
                    y_intersect = y1 + t * dy
                    if square_top <= y_intersect <= square_bottom:
                        return True

        if dy != 0:
            # 检查与上边和下边的相交
            t_top = (square_top - y1) / dy
            t_bottom = (square_bottom - y1) / dy

            for t in [t_top, t_bottom]:
                if 0 <= t <= 1:
                    x_intersect = x1 + t * dx
                    if square_left <= x_intersect <= square_right:
                        return True

        return False
        
    def add_safe_landing_path(self):
        """添加安全的45度降落路径，允许返回时走重复航点"""
        if not self.waypoints:
            return False

        # 记录返回路径的起始索引
        self.return_path_start_index = len(self.waypoints) - 1

        last_point = self.waypoints[-1]
        red_col, red_row = self.red_point

        # 检查当前最后一个点是否已经是降落点
        if last_point == self.red_point:
            # 如果最后一个点已经是降落点，直接返回成功（起降点相同的情况）
            return True

        # 检查当前最后一个点是否已经在红点的45度对角线上
        last_col, last_row = last_point

        # 如果已经在45度对角线上且路径安全，直接降落
        if (abs(last_col - red_col) == abs(last_row - red_row) and
            not self.path_crosses_forbidden_zone(last_point, self.red_point)):
            self.waypoints.append(self.red_point)
            return True
        
        # 寻找能形成45度角且不经过禁区的降落点
        landing_candidates = []
        
        # 从红点向四个对角线方向寻找合适的降落点
        directions = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # 左上、右上、左下、右下
        
        for dx, dy in directions:
            for distance in range(1, max(self.grid_cols, self.grid_rows)):
                approach_col = red_col + dx * distance
                approach_row = red_row + dy * distance
                
                if (0 <= approach_col < self.grid_cols and 
                    0 <= approach_row < self.grid_rows and
                    (approach_col, approach_row) not in self.forbidden_zones):
                    
                    # 检查从降落点到红点的45度路径是否安全
                    if not self.path_crosses_forbidden_zone((approach_col, approach_row), self.red_point):
                        landing_candidates.append((approach_col, approach_row, distance))
        
        if landing_candidates:
            # 选择距离最近的合适降落点
            landing_candidates.sort(key=lambda x: x[2])
            approach_point = (landing_candidates[0][0], landing_candidates[0][1])
            
            # 检查从当前位置到降落点的直接路径是否安全
            if not self.path_crosses_forbidden_zone(last_point, approach_point):
                # 直接路径安全，添加降落点
                if approach_point != last_point:
                    self.waypoints.append(approach_point)
                self.waypoints.append(self.red_point)
                return True
            else:
                # 直接路径不安全，允许通过重复航点绕行到降落点
                detour_path = self.find_detour_to_landing_point_with_repeats(last_point, approach_point)
                if detour_path:
                    # 添加绕行路径（允许重复航点）
                    for point in detour_path:
                        if point != last_point:  # 避免重复添加当前位置
                            self.waypoints.append(point)
                    # 最后降落到红点
                    self.waypoints.append(self.red_point)
                    return True
        
        # 如果找不到安全的45度降落路径，尝试直接绕行到红点
        if not self.path_crosses_forbidden_zone(last_point, self.red_point):
            self.waypoints.append(self.red_point)
            return True
        else:
            # 需要绕行到红点，允许重复航点
            detour_path = self.find_detour_to_landing_point_with_repeats(last_point, self.red_point)
            if detour_path:
                for point in detour_path:
                    if point != last_point:
                        self.waypoints.append(point)
                return True
            
        return False
        
    def find_detour_to_landing_point(self, start, target):
        """寻找从起点到目标点的绕行路径，避开禁飞区"""
        # 尝试不同的绕行策略
        detour_strategies = [
            self.try_corner_detour,
            self.try_edge_detour,
            self.try_wide_detour
        ]
        
        for strategy in detour_strategies:
            path = strategy(start, target)
            if path:
                return path
        return None
        
    def try_corner_detour(self, start, target):
        """尝试通过角落点绕行"""
        start_col, start_row = start
        target_col, target_row = target
        
        # 尝试两个角落点：(start_col, target_row) 和 (target_col, start_row)
        corners = [(start_col, target_row), (target_col, start_row)]
        
        for corner in corners:
            if (0 <= corner[0] < self.grid_cols and 
                0 <= corner[1] < self.grid_rows and
                corner not in self.forbidden_zones and
                corner != self.red_point):
                
                # 检查通过角落点的路径是否安全
                if (not self.path_crosses_forbidden_zone(start, corner) and
                    not self.path_crosses_forbidden_zone(corner, target)):
                    return [corner, target]
        return None
        
    def try_edge_detour(self, start, target):
        """尝试通过边缘点绕行"""
        # 尝试沿着网格边缘寻找安全路径
        edge_points = []
        
        # 添加边缘点
        for col in range(self.grid_cols):
            edge_points.extend([(col, 0), (col, self.grid_rows-1)])
        for row in range(self.grid_rows):
            edge_points.extend([(0, row), (self.grid_cols-1, row)])
        
        # 移除重复点和禁区点
        edge_points = list(set(edge_points))
        edge_points = [p for p in edge_points if p not in self.forbidden_zones and p != self.red_point]
        
        # 按距离排序
        edge_points.sort(key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
        
        for edge_point in edge_points:
            # 检查通过边缘点的路径是否安全
            if (not self.path_crosses_forbidden_zone(start, edge_point) and
                not self.path_crosses_forbidden_zone(edge_point, target)):
                return [edge_point, target]
        return None
        
    def try_wide_detour(self, start, target):
        """尝试更大范围的绕行路径"""
        # 在更大范围内寻找中间点
        for distance in range(2, max(self.grid_cols, self.grid_rows)):
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    if abs(dx) + abs(dy) == distance:  # 曼哈顿距离为distance的点
                        intermediate_col = start[0] + dx
                        intermediate_row = start[1] + dy
                        intermediate = (intermediate_col, intermediate_row)
                        
                        if (0 <= intermediate_col < self.grid_cols and 
                            0 <= intermediate_row < self.grid_rows and
                            intermediate not in self.forbidden_zones and
                            intermediate != self.red_point):
                            
                            # 检查通过中间点的路径是否安全
                            if (not self.path_crosses_forbidden_zone(start, intermediate) and
                                not self.path_crosses_forbidden_zone(intermediate, target)):
                                return [intermediate, target]
        return None

    def test_coordinate_conversion(self):
        """测试坐标转换的正确性"""
        print("=== 坐标转换测试 ===")
        print(f"网格规格: {self.grid_cols}×{self.grid_rows} = {self.grid_cols*self.grid_rows}个方格")
        print(f"方格大小: {self.cell_size*100}cm × {self.cell_size*100}cm")
        print(f"高度设置: 起降={self.takeoff_landing_height}m, 巡查={self.survey_height}m")
        print(f"红点位置 (起降点): {self.red_point}")
        print(f"原点设置: col={self.origin_col}, row={self.origin_row}")

        # 测试关键点的坐标转换
        test_points = [
            self.red_point,  # 红点应该是(0,0,0)
            (0, 0),          # B7 A1 - 左上角
            (8, 0),          # B7 A9 - 右上角
            (0, 6),          # B1 A1 - 左下角
        ]

        print("\n基础坐标转换（默认高度）:")
        for col, row in test_points:
            coord = self.position_to_coord(col, row)
            global_x, global_y, global_z = self.grid_to_global_coords(col, row)
            print(f"{coord}: 网格({col},{row}) -> 全局({global_x:.1f}m,{global_y:.1f}m,{global_z:.1f}m)")

        print("\n航点高度测试（模拟5个航点）:")
        for i in range(5):
            col, row = self.red_point  # 使用红点位置测试
            coord = self.position_to_coord(col, row)
            global_x, global_y, global_z = self.grid_to_global_coords(col, row, i, 5)
            waypoint_type = "起飞" if i == 0 else "降落" if i == 4 else "巡查"
            print(f"航点{i+1}({waypoint_type}): {coord} -> 全局({global_x:.1f}m,{global_y:.1f}m,{global_z:.2f}m)")

        print("=== 测试完成 ===")

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 保存航线动作（传统方式）
        save_action = QAction('保存航线到文件...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('将航线保存到指定的YAML文件')
        save_action.triggered.connect(self.save_waypoints)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('退出应用程序')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        # 关于动作
        about_action = QAction('关于', self)
        about_action.setStatusTip('关于无人机喷涂系统')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        """显示关于对话框"""
        about_text = (
            "无人机喷涂系统地面站 v2.0\n\n"
            "功能特性：\n"
            "• 63个50cm×50cm方格网格\n"
            "• 混合模式路径规划\n"
            "• 沿边缘优先返航\n"
            "• ROS topic航点传输\n"
            "• 差异化航点高度设置\n\n"
            "坐标系：以红点为原点\n"
            "• X轴：向前 (B1→B7)\n"
            "• Y轴：向左 (A9→A1)\n"
            "• Z轴：向上\n\n"
            "高度设置：\n"
            "• 起降高度：0.0米\n"
            "• 巡查高度：1.22米"
        )
        QMessageBox.about(self, "关于", about_text)

    def publish_waypoints_to_ros(self, waypoints_data):
        """将航点数据发布到ROS topic"""
        try:
            # 将YAML数据转换为JSON字符串发布
            json_data = json.dumps(waypoints_data, ensure_ascii=False, indent=2)

            # 创建ROS消息
            msg = String()
            msg.data = json_data

            # 发布消息
            self.waypoint_publisher.publish(msg)

            print(f"航点数据已发布到ROS topic: /wildlife_survey/waypoints")
            print(f"发布数据大小: {len(json_data)} 字符")

        except Exception as e:
            print(f"发布ROS消息失败: {e}")
            QMessageBox.warning(self, "警告", f"发布ROS消息失败: {e}")

    def upload_waypoints(self):
        """上传航线到ROS系统"""
        if not self.waypoints:
            QMessageBox.warning(self, "警告", "请先规划路径！")
            return

        # 显示进度条并开始上传过程
        self.start_upload_process()

    def start_upload_process(self):
        """开始上传过程，显示进度条"""
        # 显示进度条
        self.upload_progress.setVisible(True)
        self.upload_progress.setValue(0)
        self.upload_progress.setFormat("准备上传...")

        # 更新状态栏
        self.statusBar().showMessage("正在上传航线...")

        # 创建定时器控制进度条
        self.upload_timer = QTimer()
        self.upload_timer.timeout.connect(self.update_upload_progress)
        self.upload_step = 0
        self.upload_timer.start(100)  # 每100ms更新一次

    def update_upload_progress(self):
        """更新上传进度"""
        self.upload_step += 1
        progress = min(self.upload_step * 100 // 30, 100)  # 3秒 = 30步

        # 更新进度条
        self.upload_progress.setValue(progress)

        # 更新进度文本
        if progress < 30:
            self.upload_progress.setFormat("生成航点数据...")
        elif progress < 60:
            self.upload_progress.setFormat("发布到ROS topic...")
        elif progress < 90:
            self.upload_progress.setFormat("保存本地文件...")
        else:
            self.upload_progress.setFormat("上传完成!")

        # 在不同阶段执行实际操作
        if progress == 30:
            # 第1秒：生成数据
            self.prepare_waypoint_data()
        elif progress == 60:
            # 第2秒：发布ROS
            self.publish_to_ros()
        elif progress == 90:
            # 第3秒：保存文件
            self.save_local_file()
        elif progress >= 100:
            # 完成
            self.finish_upload()

    def prepare_waypoint_data(self):
        """准备航点数据"""
        try:
            self.waypoints_data = {
                'waypoints': [],
                'metadata': {
                    'total_points': len(self.waypoints),
                    'grid_size': f"{self.grid_cols}x{self.grid_rows}",
                    'total_cells': self.grid_cols * self.grid_rows,
                    'cell_size': {
                        'width_m': self.cell_size,
                        'height_m': self.cell_size,
                        'width_cm': self.cell_size * 100,
                        'height_cm': self.cell_size * 100
                    },
                    'coordinate_system': {
                        'origin': 'red_point' if not self.coordinate_system_initialized else 'drone_initial_position',
                        'x_axis': 'forward (B1->B7)',
                        'y_axis': 'left (A9->A1)',
                        'z_axis': 'up',
                        'units': 'meters'
                    },
                    'height_settings': {
                        'takeoff_landing_height_m': self.takeoff_landing_height,
                        'survey_height_m': self.survey_height,
                        'description': 'First and last waypoints at takeoff/landing height, others at survey height'
                    },
                    'forbidden_zones': [self.position_to_coord(col, row) for col, row in self.forbidden_zones],
                    'red_point': self.position_to_coord(*self.red_point),
                    'red_point_global': {'x': 0, 'y': 0, 'z': 0}
                }
            }
            
            # 如果坐标系已初始化，添加初始位置信息
            if self.coordinate_system_initialized and self.drone_initial_position:
                self.waypoints_data['metadata']['drone_initial_position'] = {
                    'x': self.drone_initial_position[0],
                    'y': self.drone_initial_position[1],
                    'z': self.drone_initial_position[2]
                }
                self.waypoints_data['metadata']['coordinate_offsets'] = {
                    'offset_x': self.offset_x,
                    'offset_y': self.offset_y,
                    'offset_z': self.offset_z
                }

            for i, (col, row) in enumerate(self.waypoints):
                coord = self.position_to_coord(col, row)
                global_x, global_y, global_z = self.grid_to_global_coords(col, row, i, len(self.waypoints))

                # 确定航点动作类型
                if i == 0:
                    action = 'takeoff'
                elif i == len(self.waypoints) - 1:
                    action = 'land'
                else:
                    action = 'survey'

                # 如果是第一个航点且坐标系已初始化，使用无人机初始位置作为起点
                if i == 0 and self.coordinate_system_initialized and self.drone_initial_position:
                    waypoint = {
                        'id': i + 1,
                        'coordinate': 'DRONE_POSITION',
                        'grid_position': {'col': col, 'row': row},
                        'global_position': {'x': -self.offset_x, 'y': -self.offset_y, 'z': self.survey_height},
                        'action': 'takeoff',
                        'height_info': {
                            'is_takeoff_landing': True,
                            'height_m': self.survey_height
                        },
                        'original_drone_position': {
                            'x': self.drone_initial_position[0],
                            'y': self.drone_initial_position[1],
                            'z': self.drone_initial_position[2]
                        }
                    }
                else:
                    waypoint = {
                        'id': i + 1,
                        'coordinate': coord,
                        'grid_position': {'col': col, 'row': row},
                        'global_position': {'x': global_x, 'y': global_y, 'z': global_z},
                        'action': action,
                        'height_info': {
                            'is_takeoff_landing': i == 0 or i == len(self.waypoints) - 1,
                            'height_m': global_z
                        }
                    }
                self.waypoints_data['waypoints'].append(waypoint)

        except Exception as e:
            print(f"准备数据时出错: {e}")
            self.upload_error = str(e)

    def publish_to_ros(self):
        """发布数据到ROS topic"""
        try:
            self.publish_waypoints_to_ros(self.waypoints_data)
        except Exception as e:
            print(f"发布ROS数据时出错: {e}")
            self.upload_error = str(e)

    def save_local_file(self):
        """保存本地文件"""
        try:
            # 使用默认文件名
            filename = "waypoints.yaml"
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(self.waypoints_data, f, default_flow_style=False, allow_unicode=True)
            self.saved_filename = filename
        except Exception as e:
            print(f"保存本地文件时出错: {e}")
            self.upload_error = str(e)

    def finish_upload(self):
        """完成上传过程"""
        # 停止定时器
        self.upload_timer.stop()

        # 隐藏进度条
        QTimer.singleShot(1000, lambda: self.upload_progress.setVisible(False))

        # 检查是否有错误
        if hasattr(self, 'upload_error'):
            self.statusBar().showMessage("上传失败", 5000)
            QMessageBox.critical(self, "上传失败", f"上传过程中出现错误：\n{self.upload_error}")
            delattr(self, 'upload_error')
            return

        # 更新状态栏
        self.statusBar().showMessage("航线上传成功", 5000)

        # 显示成功信息
        self.show_upload_success()

    def show_upload_success(self):
        """显示上传成功信息"""
        total_distance = self.calculate_total_distance()
        takeoff_landing_count = 2 if len(self.waypoints) > 1 else 1
        survey_count = len(self.waypoints) - takeoff_landing_count

        success_info = (
            f"航线上传成功！\n\n"
            f"📁 本地文件：{getattr(self, 'saved_filename', 'waypoints.yaml')}\n"
            f"📡 ROS Topic：/wildlife_survey/waypoints\n\n"
            f"网格信息：\n"
            f"• 方格数量：{self.grid_cols}×{self.grid_rows} = {self.grid_cols*self.grid_rows}个\n"
            f"• 方格大小：{self.cell_size*100:.0f}cm × {self.cell_size*100:.0f}cm\n"
            f"• 总航点：{len(self.waypoints)}个\n"
            f"• 总距离：{total_distance:.1f}米\n\n"
            f"高度信息：\n"
            f"• 起降航点：{takeoff_landing_count}个，高度{self.takeoff_landing_height}米\n"
            f"• 巡查航点：{survey_count}个，高度{self.survey_height}米\n\n"
            f"坐标系信息：\n"
            f"• 原点：红点起降点 (0,0,0)\n"
            f"• X轴：向前 (B1→B7)\n"
            f"• Y轴：向左 (A9→A1)\n"
            f"• Z轴：向上\n"
            f"• 单位：米"
        )
        QMessageBox.information(self, "上传成功", success_info)

    def save_waypoints(self):
        """保存航线到YAML文件（保留原功能）"""
        if not self.waypoints:
            QMessageBox.warning(self, "警告", "请先规划路径！")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存航线", "waypoints.yaml", "YAML files (*.yaml)")
            
        if filename:
            try:
                waypoints_data = {
                    'waypoints': [],
                    'metadata': {
                        'total_points': len(self.waypoints),
                        'grid_size': f"{self.grid_cols}x{self.grid_rows}",
                        'total_cells': self.grid_cols * self.grid_rows,
                        'cell_size': {
                            'width_m': self.cell_size,
                            'height_m': self.cell_size,
                            'width_cm': self.cell_size * 100,
                            'height_cm': self.cell_size * 100
                        },
                        'coordinate_system': {
                            'origin': 'red_point',
                            'x_axis': 'forward (B1->B7)',
                            'y_axis': 'left (A9->A1)',
                            'z_axis': 'up',
                            'units': 'meters'
                        },
                        'height_settings': {
                            'takeoff_landing_height_m': self.takeoff_landing_height,
                            'survey_height_m': self.survey_height,
                            'description': 'First and last waypoints at takeoff/landing height, others at survey height'
                        },
                        'forbidden_zones': [self.position_to_coord(col, row) for col, row in self.forbidden_zones],
                        'red_point': self.position_to_coord(*self.red_point),
                        'red_point_global': {'x': 0, 'y': 0, 'z': 0}
                    }
                }

                for i, (col, row) in enumerate(self.waypoints):
                    coord = self.position_to_coord(col, row)
                    global_x, global_y, global_z = self.grid_to_global_coords(col, row, i, len(self.waypoints))

                    # 确定航点动作类型
                    if i == 0:
                        action = 'takeoff'
                    elif i == len(self.waypoints) - 1:
                        action = 'land'
                    else:
                        action = 'survey'

                    waypoint = {
                        'id': i + 1,
                        'coordinate': coord,
                        'grid_position': {'col': col, 'row': row},
                        'global_position': {'x': global_x, 'y': global_y, 'z': global_z},
                        'action': action,
                        'height_info': {
                            'is_takeoff_landing': i == 0 or i == len(self.waypoints) - 1,
                            'height_m': global_z
                        }
                    }
                    waypoints_data['waypoints'].append(waypoint)
                    
                with open(filename, 'w', encoding='utf-8') as f:
                    yaml.dump(waypoints_data, f, default_flow_style=False, allow_unicode=True)

                # 发布航点数据到ROS topic
                self.publish_waypoints_to_ros(waypoints_data)

                # 显示保存成功信息，包含坐标系说明
                total_distance = self.calculate_total_distance()
                takeoff_landing_count = 2 if len(self.waypoints) > 1 else 1
                survey_count = len(self.waypoints) - takeoff_landing_count

                coord_info = (
                    f"航线已保存到 {filename}\n"
                    f"数据已发布到ROS topic: /wildlife_survey/waypoints\n\n"
                    f"网格信息：\n"
                    f"• 方格数量：{self.grid_cols}×{self.grid_rows} = {self.grid_cols*self.grid_rows}个\n"
                    f"• 方格大小：{self.cell_size*100:.0f}cm × {self.cell_size*100:.0f}cm\n"
                    f"• 总航点：{len(self.waypoints)}个\n"
                    f"• 总距离：{total_distance:.1f}米\n\n"
                    f"高度信息：\n"
                    f"• 起降航点：{takeoff_landing_count}个，高度{self.takeoff_landing_height}米\n"
                    f"• 巡查航点：{survey_count}个，高度{self.survey_height}米\n\n"
                    f"坐标系信息：\n"
                    f"• 原点：红点起降点 (0,0,0)\n"
                    f"• X轴：向前 (B1→B7)\n"
                    f"• Y轴：向左 (A9→A1)\n"
                    f"• Z轴：向上\n"
                    f"• 单位：米"
                )
                QMessageBox.information(self, "成功", coord_info)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                
    def toggle_mission(self):
        """切换任务状态"""
        self.mission_active = not self.mission_active
        
        if self.mission_active:
            # 开始任务
            self.start_btn.setText("停止")
            self.start_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 15px; }")
            self.status_label.setText("状态: 任务进行中")
            
            # 重置任务完成标志
            self.mission_completed = False
            self.drone_returned_to_origin = False
            
            # 任务开始时生成新的任务ID（如果之前有数据且未完成，先询问是否保存）
            if self.wildlife_detections:
                reply = QMessageBox.question(self, '开始新任务', 
                    '有未保存的无人机喷涂检测数据，是否保存后再开始新任务？',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
                if reply == QMessageBox.Yes:
                    self.save_wildlife_data()
                    self.wildlife_detections.clear()
                    self.update_wildlife_display()
            
            # 生成新的任务ID
            self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            rospy.loginfo(f"开始新任务，ID: {self.mission_id}")
            
            # 清空轨迹，准备新任务
            self.drone_trajectory.clear()
            self.map_widget.update()
        else:
            # 停止任务
            self.start_btn.setText("开始")
            self.start_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 15px; }")
            self.status_label.setText("状态: 待机")
            
            # 任务结束时，自动保存动物检测数据
            if self.wildlife_detections:
                try:
                    self.save_wildlife_data()
                    QMessageBox.information(self, "任务完成", f"任务已停止，无人机喷涂检测数据已自动保存至:\n{os.path.join(self.history_dir, self.mission_id+'.json')}")
                except Exception as e:
                    rospy.logerr(f"自动保存动物检测数据失败: {e}")
                    QMessageBox.warning(self, "数据保存警告", f"自动保存数据时出现错误: {str(e)}")
            else:
                QMessageBox.information(self, "任务完成", "任务已停止，未检测到任何无人机喷涂数据")
    
    def reset_mission(self):
        """重置任务"""
        self.mission_active = False
        self.start_btn.setText("开始")
        self.start_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 15px; }")
        self.status_label.setText("状态: 待机")
        
        # 清空航点
        self.waypoints.clear()
        self.return_path_start_index = -1
        self.waypoint_count_label.setText("路径长度: 0 格")
        
        # 清空禁区
        self.forbidden_zones.clear()
        self.update_forbidden_status()
        
        # 更新显示
        self.map_widget.update()
        
        QMessageBox.information(self, "信息", "任务已重置")
        
    def publish_command(self):
        """发布命令到ROS话题"""
        command = Int32()
        command.data = 1 if self.mission_active else 0
        self.command_pub.publish(command)

    def update_wildlife_display(self):
        """更新无人机喷涂检测信息显示"""
        if not hasattr(self, 'wildlife_info_text'):
            return
            
        # 构建显示文本
        html_text = "<html><body>"
        html_text += "<h3 style='font-size:12px;margin:5px 0;'>无人机喷涂检测结果</h3>"
        
        # 检测是否是历史记录
        if self.loaded_history_id is not None:
            html_text += f"<p style='font-size:10px;color:#2196F3;margin:5px 0;'>当前显示: 历史记录 {self.loaded_history_id}</p>"
        
        html_text += "<table border='0' cellspacing='2' cellpadding='2' style='font-size:10px;'>"
        html_text += "<tr><th>方格</th><th>动物种类</th><th>数量</th></tr>"
        
        # 对检测记录按方格位置排序
        sorted_detections = sorted(self.wildlife_detections.items(), 
                                  key=lambda x: x[1]['grid_coord'])
        
        # 生成显示内容
        for grid_pos, data in sorted_detections:
            grid_name = data['grid_coord']
            for animal_name, count in data['animals'].items():
                html_text += f"<tr><td>{grid_name}</td><td>{animal_name}</td><td align='center'>{count}只</td></tr>"
        
        if not self.wildlife_detections:
            html_text += "<tr><td colspan='3' align='center' style='color:#666;'>暂无检测数据</td></tr>"
        
        html_text += "</table></body></html>"
        
        # 更新显示
        self.wildlife_info_text.setHtml(html_text)

    def create_wildlife_panel(self):
        """创建无人机喷涂检测信息显示面板"""
        group_box = QGroupBox("无人机喷涂检测信息")
        group_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        
        layout = QVBoxLayout(group_box)
        
        # 创建顶部操作栏
        top_bar = QHBoxLayout()
        
        # 创建历史记录按钮
        self.history_btn = QPushButton("显示历史记录")
        self.history_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 5px;
                font-size: 10px; 
            }
        """)
        self.history_btn.clicked.connect(self.toggle_history_panel)
        top_bar.addWidget(self.history_btn)
        
        # 添加保存按钮
        save_btn = QPushButton("保存检测数据")
        save_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 5px;
                font-size: 10px; 
            }
        """)
        save_btn.clicked.connect(self.save_wildlife_data)
        top_bar.addWidget(save_btn)
        
        layout.addLayout(top_bar)
        
        # 创建信息标签
        info_label = QLabel("检测到的无人机喷涂将在此显示")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        layout.addWidget(info_label)
        
        # 创建文本显示区域
        self.wildlife_info_text = QTextBrowser()
        self.wildlife_info_text.setMinimumHeight(200)
        self.wildlife_info_text.setStyleSheet("""
            QTextBrowser {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                font-family: Arial, sans-serif;
                font-size: 11px;
            }
        """)
        
        # 设置初始HTML内容
        initial_html = """
        <html>
        <body>
        <h3 style='font-size:12px;'>无人机喷涂检测结果</h3>
        <p style='color:#666;font-size:10px;'>当无人机到达方格中心时，将显示检测到的无人机喷涂信息</p>
        </body>
        </html>
        """
        self.wildlife_info_text.setHtml(initial_html)
        
        layout.addWidget(self.wildlife_info_text)
        
        return group_box
        
    def create_history_panel(self):
        """创建历史记录面板（从右侧滑出）"""
        # 创建历史记录面板
        self.history_panel = QWidget(self)
        self.history_panel.setFixedWidth(300)
        self.history_panel.setStyleSheet("background-color: white; border-left: 1px solid #ccc;")
        
        # 设置面板位置（在窗口右侧）
        self.history_panel.setGeometry(self.width(), 0, 300, self.height())
        
        # 创建布局
        layout = QVBoxLayout(self.history_panel)
        
        # 创建标题
        title_label = QLabel("历史记录")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # 创建历史记录列表
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                font-size: 11px;
                border: 1px solid #ddd;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e0f0ff;
                color: black;
            }
        """)
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        layout.addWidget(self.history_list)
        
        # 创建底部按钮区域
        button_layout = QHBoxLayout()
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.toggle_history_panel)
        button_layout.addWidget(close_btn)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(lambda: (setattr(self, 'history_records', self.load_history_list()), self.update_history_list()))
        button_layout.addWidget(refresh_btn)
        
        # 添加重置按钮
        reset_btn = QPushButton("重置地面站")
        reset_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF5722; 
                color: white; 
                font-weight: bold; 
                padding: 5px;
            }
        """)
        reset_btn.clicked.connect(self.reset_ground_station)
        layout.addWidget(reset_btn)
        
        layout.addLayout(button_layout)
        
        # 初始状态为隐藏
        self.history_panel.hide()
        
    def resizeEvent(self, event):
        """重写窗口大小改变事件，确保历史面板位置正确"""
        super().resizeEvent(event)
        if hasattr(self, 'history_panel'):
            if self.history_panel_visible:
                # 如果历史面板可见，显示在窗口内
                self.history_panel.setGeometry(self.width() - 300, 0, 300, self.height())
            else:
                # 如果历史面板隐藏，位置在窗口外
                self.history_panel.setGeometry(self.width(), 0, 300, self.height())
                
    def toggle_history_panel(self):
        """切换历史记录面板的显示状态"""
        if self.history_panel_visible:
            # 隐藏历史面板
            self.history_panel.setGeometry(self.width(), 0, 300, self.height())
            self.history_panel_visible = False
            self.history_btn.setText("显示历史记录")
        else:
            # 显示历史面板
            self.history_panel.setGeometry(self.width() - 300, 0, 300, self.height())
            self.history_panel.show()
            self.history_panel_visible = True
            self.history_btn.setText("隐藏历史记录")
            # 更新历史记录列表
            self.update_history_list()
    
    def on_history_item_clicked(self, item):
        """处理历史记录项点击事件"""
        record_id = item.data(Qt.UserRole)
        if record_id:
            if self.load_wildlife_history(record_id):
                rospy.loginfo(f"已加载历史记录: {record_id}")
                # 更新列表选中状态
                self.update_history_list()

    def load_history_list(self):
        """加载历史记录列表"""
        history_records = []
        try:
            if os.path.exists(self.history_dir):
                # 获取所有历史记录文件
                for filename in sorted(os.listdir(self.history_dir), reverse=True):
                    if filename.endswith('.json'):
                        record_id = os.path.splitext(filename)[0]
                        file_path = os.path.join(self.history_dir, filename)
                        
                        try:
                            # 尝试读取记录文件获取元数据
                            with open(file_path, 'r', encoding='utf-8') as f:
                                record_data = json.load(f)
                                
                            # 提取记录信息
                            timestamp = record_id
                            detection_count = 0
                            animal_count = 0
                            
                            if 'detections' in record_data:
                                detection_count = len(record_data['detections'])
                                # 统计动物总数
                                for grid_data in record_data['detections'].values():
                                    if 'animals' in grid_data:
                                        for count in grid_data['animals'].values():
                                            animal_count += count
                            
                            # 格式化时间显示
                            try:
                                if '_' in timestamp:
                                    date_part, time_part = timestamp.split('_')
                                    formatted_time = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                                else:
                                    formatted_time = timestamp
                            except:
                                formatted_time = timestamp
                            
                            # 添加到记录列表
                            history_records.append({
                                'id': record_id,
                                'file_path': file_path,
                                'time': formatted_time,
                                'detection_count': detection_count,
                                'animal_count': animal_count
                            })
                        except Exception as e:
                            rospy.logwarn(f"读取历史记录文件 {filename} 失败: {e}")
            
            return history_records
        except Exception as e:
            rospy.logerr(f"加载历史记录列表失败: {e}")
            return []
    
    def save_wildlife_data(self):
        """保存无人机喷涂检测数据"""
        if not self.wildlife_detections:
            rospy.loginfo("没有无人机喷涂检测数据需要保存")
            return
        
        try:
            # 准备保存数据
            save_data = {
                'mission_id': self.mission_id,
                'timestamp': datetime.now().isoformat(),
                'detections': {}
            }
            
            # 转换数据格式
            for grid_pos, data in self.wildlife_detections.items():
                grid_key = f"{grid_pos[0]}_{grid_pos[1]}"
                save_data['detections'][grid_key] = {
                    'grid_position': {'col': grid_pos[0], 'row': grid_pos[1]},
                    'grid_coord': data['grid_coord'],
                    'detection_time': data['detection_time'],
                    'animals': data['animals']
                }
            
            # 保存到文件
            filename = f"{self.mission_id}.json"
            filepath = os.path.join(self.history_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            rospy.loginfo(f"无人机喷涂检测数据已保存至: {filepath}")
            
            # 更新历史记录列表
            self.history_records = self.load_history_list()
            # 如果历史面板可见，刷新显示
            if hasattr(self, 'history_list') and self.history_panel_visible:
                self.update_history_list()
                
        except Exception as e:
            rospy.logerr(f"保存无人机喷涂数据失败: {e}")
    
    def load_wildlife_history(self, record_id):
        """加载历史无人机喷涂检测数据"""
        try:
            filepath = os.path.join(self.history_dir, f"{record_id}.json")
            if not os.path.exists(filepath):
                rospy.logwarn(f"历史记录文件不存在: {filepath}")
                return False
                
            # 读取历史数据
            with open(filepath, 'r', encoding='utf-8') as f:
                record_data = json.load(f)
            
            # 清空当前检测记录
            self.wildlife_detections = {}
            
            # 加载历史检测记录
            if 'detections' in record_data:
                for grid_key, data in record_data['detections'].items():
                    col, row = map(int, grid_key.split('_'))
                    grid_pos = (col, row)
                    
                    self.wildlife_detections[grid_pos] = {
                        'detection_time': data.get('detection_time', 0),
                        'animals': data.get('animals', {}),
                        'grid_coord': data.get('grid_coord', '')
                    }
            
            # 更新已加载的历史记录ID
            self.loaded_history_id = record_id
            
            # 更新显示
            self.update_wildlife_display()
            self.map_widget.update()
            
            return True
        
        except Exception as e:
            rospy.logerr(f"加载历史记录失败: {e}")
            return False
    
    def update_history_list(self):
        """更新历史记录列表显示"""
        if not hasattr(self, 'history_list'):
            return
            
        # 清空列表
        self.history_list.clear()
        
        # 添加记录项
        for record in self.history_records:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, record['id'])
            
            # 构建显示文本
            text = f"{record['time']}\n"
            text += f"检测区域: {record['detection_count']}处, 动物总数: {record['animal_count']}只"
            
            item.setText(text)
            
            # 设置选中状态
            if record['id'] == self.loaded_history_id:
                item.setSelected(True)
                item.setBackground(QColor(230, 245, 255))
            
            self.history_list.addItem(item)
    
    def reset_ground_station(self):
        """重置地面站状态"""
        # 如果有未保存的数据，提示用户保存
        if self.wildlife_detections and not self.mission_completed:
            reply = QMessageBox.question(self, '保存数据', 
                '有未保存的无人机喷涂检测数据，是否保存？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                self.save_wildlife_data()
        
        # 重置任务状态
        self.mission_active = False
        self.mission_completed = False
        self.drone_returned_to_origin = False
        
        # 重置命令状态
        command = Int32()
        command.data = 0  # 0表示停止任务
        self.command_pub.publish(command)
        
        # 更新UI状态
        self.start_btn.setText("开始")
        self.start_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 15px; }")
        self.status_label.setText("状态: 待机")
        
        # 清空航点
        self.waypoints.clear()
        self.return_path_start_index = -1
        self.waypoint_count_label.setText("路径长度: 0 格")
        
        # 清空禁区
        self.forbidden_zones.clear()
        self.update_forbidden_status()
        
        # 清空无人机喷涂检测数据（创建新的任务ID）
        self.wildlife_detections.clear()
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 重置历史记录ID
        self.loaded_history_id = None
        self.update_wildlife_display()
        
        # 清空无人机轨迹
        self.drone_trajectory.clear()
        
        # 加载新的历史记录列表
        self.history_records = self.load_history_list()
        if self.history_panel_visible:
            self.update_history_list()
        
        # 更新地图显示
        self.map_widget.update()
        
        # 通知用户重置完成
        QMessageBox.information(self, "信息", "地面站已重置，准备开始新任务")
        
        # 关闭历史记录面板
        if self.history_panel_visible:
            self.toggle_history_panel()

class MapWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMinimumSize(600, 400)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算绘制区域
        margin = 50
        draw_width = self.width() - 2 * margin
        draw_height = self.height() - 2 * margin
        
        # 计算网格大小
        cell_width = draw_width / self.parent.grid_cols
        cell_height = draw_height / self.parent.grid_rows
        
        # 绘制网格
        painter.setPen(QPen(Qt.black, 1))
        for i in range(self.parent.grid_cols + 1):
            x = margin + i * cell_width
            painter.drawLine(int(x), margin, int(x), int(margin + draw_height))
            
        for i in range(self.parent.grid_rows + 1):
            y = margin + i * cell_height
            painter.drawLine(margin, int(y), int(margin + draw_width), int(y))
            
        # 绘制坐标标签
        painter.setPen(QPen(Qt.black, 1))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        
        # 列标签 (A1-A9) - 在底部水平显示
        for i in range(self.parent.grid_cols):
            x = margin + (i + 0.5) * cell_width - 8
            y = margin + draw_height + 20
            painter.drawText(int(x), int(y), f"A{i + 1}")
            
        # 行标签 (B7-B1) - 在左侧垂直显示，从上到下为B7到B1
        for i in range(self.parent.grid_rows):
            x = margin - 25
            y = margin + (i + 0.5) * cell_height + 5
            painter.drawText(int(x), int(y), f"B{7 - i}")
            
        # 绘制禁区
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.setPen(QPen(Qt.darkRed, 2))
        for col, row in self.parent.forbidden_zones:
            x = margin + col * cell_width
            y = margin + row * cell_height  # 直接使用row，因为索引0对应A9（顶部），索引8对应A1（底部）
            painter.drawRect(int(x), int(y), int(cell_width), int(cell_height))
        
        # 绘制检测到无人机喷涂的方格 - 使用淡绿色填充，并添加动物图标
        if hasattr(self.parent, 'wildlife_detections') and self.parent.wildlife_detections:
            for grid_pos, data in self.parent.wildlife_detections.items():
                col, row = grid_pos
                x = margin + col * cell_width
                y = margin + row * cell_height
                
                # 使用淡绿色半透明填充表示有动物检测
                painter.setBrush(QBrush(QColor(100, 200, 100, 80), Qt.SolidPattern))
                painter.setPen(QPen(QColor(50, 150, 50), 2))
                painter.drawRect(int(x), int(y), int(cell_width), int(cell_height))
                
                # 在方格中央添加一个动物图标标记
                center_x = x + cell_width / 2
                center_y = y + cell_height / 2
                
                # 绘制一个小图标表示检测到动物
                icon_size = min(cell_width, cell_height) * 0.4
                painter.setBrush(QBrush(QColor(50, 150, 50), Qt.SolidPattern))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawEllipse(int(center_x - icon_size/2), int(center_y - icon_size/2), 
                                   int(icon_size), int(icon_size))
                
                # 在方格内显示动物总数
                total_animals = sum(data['animals'].values())
                painter.setPen(QPen(Qt.white, 1))
                font = QFont()
                font.setBold(True)
                font.setPointSize(10)
                painter.setFont(font)
                text_width = painter.fontMetrics().width(str(total_animals))
                painter.drawText(int(center_x - text_width/2), int(center_y + 5), str(total_animals))
            
        # 绘制红点(起降点)
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.setPen(QPen(Qt.darkRed, 3))
        red_col, red_row = self.parent.red_point
        x = margin + (red_col + 0.5) * cell_width
        y = margin + (red_row + 0.5) * cell_height  # 直接使用red_row
        
        # 在调试信息中输出红点位置的坐标信息，便于确认
        if hasattr(self.parent, 'origin_col') and hasattr(self.parent, 'origin_row'):
            # 确保红点位置就是原点(0,0,0)对应的网格位置
            self.parent.origin_col = red_col
            self.parent.origin_row = red_row
        
        # 绘制红点
        painter.drawEllipse(int(x-10), int(y-10), 20, 20)
        
        # 绘制航线
        if len(self.parent.waypoints) > 1:
            for i in range(len(self.parent.waypoints) - 1):
                col1, row1 = self.parent.waypoints[i]
                col2, row2 = self.parent.waypoints[i + 1]
                
                x1 = margin + (col1 + 0.5) * cell_width
                y1 = margin + (row1 + 0.5) * cell_height  # 直接使用row1
                x2 = margin + (col2 + 0.5) * cell_width
                y2 = margin + (row2 + 0.5) * cell_height  # 直接使用row2
                
                # 判断是巡查路径还是返回路径
                if (self.parent.return_path_start_index >= 0 and 
                    i >= self.parent.return_path_start_index):
                    # 返回路径用紫色
                    painter.setPen(QPen(QColor(128, 0, 128), 3))  # 紫色
                else:
                    # 巡查路径用绿色
                    painter.setPen(QPen(Qt.green, 3))
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                
                # 为巡查路径绘制箭头
                if (self.parent.return_path_start_index < 0 or 
                    i < self.parent.return_path_start_index):
                    self.draw_arrow(painter, x1, y1, x2, y2)
                
        # 绘制航点
        painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        painter.setPen(QPen(Qt.darkGreen, 2))
        for i, (col, row) in enumerate(self.parent.waypoints):
            x = margin + (col + 0.5) * cell_width
            y = margin + (row + 0.5) * cell_height  # 直接使用row
            painter.drawEllipse(int(x-5), int(y-5), 10, 10)
            
            # 绘制航点编号
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(int(x+8), int(y+5), str(i+1))
            painter.setPen(QPen(Qt.darkGreen, 2))
            
        # 绘制无人机轨迹
        if len(self.parent.drone_trajectory) > 1:
            painter.setPen(QPen(QColor(255, 140, 0, 120), 2))  # 半透明橘色
            
            prev_point = None
            for point in self.parent.drone_trajectory:
                # 将全局坐标转换为网格坐标
                grid_col, grid_row = self.parent.global_to_grid_coords(point[0], point[1])
                
                # 转换为屏幕坐标
                screen_x = margin + (grid_col + 0.5) * cell_width
                screen_y = margin + (grid_row + 0.5) * cell_height
                
                if prev_point is not None:
                    painter.drawLine(int(prev_point[0]), int(prev_point[1]), int(screen_x), int(screen_y))
                
                prev_point = (screen_x, screen_y)
            
        # 绘制无人机当前位置（橘色圆点）
        if hasattr(self.parent, 'drone_position'):
            # 将全局坐标转换为网格坐标
            drone_x, drone_y, drone_z = self.parent.drone_position
            
            # 处理特殊情况：如果无人机位于原点(0,0,0)附近
            if abs(drone_x) < 0.01 and abs(drone_y) < 0.01:
                # 使用红点位置但稍微偏移一点，确保橙色点能够显示在红点上方
                grid_col, grid_row = self.parent.red_point
                self.parent.last_displayed_grid_position = (grid_col, grid_row)
            else:
                # 正常坐标转换
                grid_col, grid_row = self.parent.global_to_grid_coords(drone_x, drone_y)
                
                # 检查距离方格中心的距离，如果大于0.1米且有上一次显示位置，则使用上一次位置
                if hasattr(self.parent, 'last_distance_to_center'):
                    distance = self.parent.last_distance_to_center
                    
                    # 更新距离标签显示
                    if hasattr(self.parent, 'distance_to_center_label'):
                        status_text = f"距离方格中心: {distance:.2f}m"
                        if distance < 0.1:
                            status_text += " (已到达)"
                            self.parent.distance_to_center_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")  # 绿色加粗
                        else:
                            self.parent.distance_to_center_label.setStyleSheet("QLabel { color: #FF8C00; }")  # 橘色
                        self.parent.distance_to_center_label.setText(status_text)
                    
                    # 显示距离信息在状态栏（可选）
                    if hasattr(self.parent, 'statusBar'):
                        status_msg = f"距离方格中心: {distance:.2f}米"
                        if distance < 0.1:
                            status_msg += " (已到达)"
                        self.parent.statusBar().showMessage(status_msg, 1000)
                    
                    # 如果距离大于0.1米，并且有上一次位置，则保持使用上一次位置
                    if distance > 0.1 and self.parent.last_displayed_grid_position is not None:
                        grid_col, grid_row = self.parent.last_displayed_grid_position
                    else:
                        # 更新为新位置
                        self.parent.last_displayed_grid_position = (grid_col, grid_row)
            
            # 转换为屏幕坐标
            screen_x = margin + (grid_col + 0.5) * cell_width
            screen_y = margin + (grid_row + 0.5) * cell_height
            
            # 更新位置标签
            self.parent.drone_position_label.setText(f"无人机位置: X={drone_x:.2f}m Y={drone_y:.2f}m Z={drone_z:.2f}m")
            
            # 绘制无人机位置（橘色圆点）
            painter.setBrush(QBrush(QColor(255, 140, 0), Qt.SolidPattern))  # 橘色
            painter.setPen(QPen(QColor(204, 85, 0), 2))  # 深橘色边框
            painter.drawEllipse(int(screen_x-8), int(screen_y-8), 16, 16)
    
    def draw_arrow(self, painter, x1, y1, x2, y2):
        """在线段上绘制箭头"""
        # 计算线段的方向向量
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
            
        # 单位方向向量
        ux = dx / length
        uy = dy / length
        
        # 箭头参数
        arrow_length = 15
        arrow_angle = math.pi / 6  # 30度
        
        # 箭头位置（线段中点）
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 计算箭头的两个端点
        cos_angle = math.cos(arrow_angle)
        sin_angle = math.sin(arrow_angle)
        
        # 箭头左端点
        left_x = mid_x - arrow_length * (ux * cos_angle + uy * sin_angle)
        left_y = mid_y - arrow_length * (uy * cos_angle - ux * sin_angle)
        
        # 箭头右端点
        right_x = mid_x - arrow_length * (ux * cos_angle - uy * sin_angle)
        right_y = mid_y - arrow_length * (uy * cos_angle + ux * sin_angle)
        
        # 绘制箭头线条（确保使用整数坐标）
        arrow_points = QPolygon()
        arrow_points.append(QPoint(int(mid_x), int(mid_y)))
        arrow_points.append(QPoint(int(left_x), int(left_y)))
        arrow_points.append(QPoint(int(right_x), int(right_y)))
        
        # 填充箭头
        painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        painter.drawPolygon(arrow_points)
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            # 计算点击位置对应的网格坐标
            margin = 50
            draw_width = self.width() - 2 * margin
            draw_height = self.height() - 2 * margin
            
            cell_width = draw_width / self.parent.grid_cols
            cell_height = draw_height / self.parent.grid_rows
            
            # 转换鼠标坐标到网格坐标
            x = event.x() - margin
            y = event.y() - margin
            
            if 0 <= x <= draw_width and 0 <= y <= draw_height:
                col = int(x / cell_width)
                row = int(y / cell_height)  # 直接使用y坐标，因为A9在顶部（索引0），A1在底部（索引8）
                
                # 确保坐标在有效范围内
                if 0 <= col < self.parent.grid_cols and 0 <= row < self.parent.grid_rows:
                    # 添加禁区
                    self.parent.add_forbidden_zone(col, row)
                    self.update()  # 重新绘制地图
             
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    try:
        window = WildlifeSurveyStation()
        window.show()
        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass