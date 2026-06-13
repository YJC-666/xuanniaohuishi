#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
玄鸟绘世 - MCP 轨迹喷涂地面站

基于 PyQt5 的无人机喷涂绘画地面站控制系统。
功能：设计加载 -> 轨迹预览 -> 喷涂控制 -> 进度监测
"""

import sys
import os
import json
import math
import yaml

import rospy
from std_msgs.msg import String, Int32, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np

# 导入 MCP 规划器
from mcp_planner import MCPPlanner


class McpPainterStation(QMainWindow):
    """MCP 喷涂地面站主窗口"""

    # 信号定义
    update_odom_signal = pyqtSignal(object)
    update_status_signal = pyqtSignal(str, int)
    update_progress_signal = pyqtSignal(int, int)
    update_coverage_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("玄鸟绘世 - MCP 喷涂地面站 v1.0")
        self.setMinimumSize(1280, 800)

        # ---- 核心数据 ----
        self.planner = MCPPlanner()
        self.waypoints = []           # 所有航点
        self.current_wp_index = 0     # 当前执行到第几个航点
        self.total_waypoints = 0
        self.mission_active = False
        self.mission_paused = False
        self.mission_completed = False

        # 无人机位置
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

        # 喷涂进度
        self.coverage_percent = 0.0
        self.completed_strokes = 0
        self.total_strokes = 0

        # 可视化
        self.trajectory_overlay = None
        self.current_preview = None

        # ---- ROS 初始化 ----
        rospy.init_node('mcp_painter_station', anonymous=True)

        # 发布者
        self.cmd_pub = rospy.Publisher('/mcp/command', String, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/mcp/trajectory', String, queue_size=1)
        self.sprayer_pub = rospy.Publisher('/mcp/sprayer', Bool, queue_size=10)

        # 订阅者
        rospy.Subscriber('/mcp/status', String, self._ros_status_cb)
        rospy.Subscriber('/mcp/coverage', String, self._ros_coverage_cb)
        rospy.Subscriber('/drone/odom', Odometry, self._ros_odom_cb)

        # ---- UI 构建 ----
        self._build_ui()
        self._connect_signals()

        # ---- 定时器 ----
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self._spin_ros)
        self.ros_timer.start(50)  # 20Hz

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)  # 10Hz

        # 状态
        self._set_status("就绪", QColor(0, 180, 0))

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # == 左侧面板: 设计加载 + 轨迹预览 ==
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 设计加载区
        load_group = QGroupBox("设计加载")
        load_layout = QHBoxLayout(load_group)
        self.design_path_edit = QLineEdit()
        self.design_path_edit.setPlaceholderText("选择 AI 设计图...")
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self._browse_design)
        self.plan_btn = QPushButton("提取轨迹")
        self.plan_btn.clicked.connect(self._plan_trajectory)
        self.plan_btn.setEnabled(False)
        load_layout.addWidget(self.design_path_edit, 1)
        load_layout.addWidget(self.browse_btn)
        load_layout.addWidget(self.plan_btn)
        left_layout.addWidget(load_group)

        # 轨迹预览画布
        preview_group = QGroupBox("轨迹预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(500, 400)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self.preview_label.setText("加载设计图后点击「提取轨迹」")
        preview_layout.addWidget(self.preview_label)
        left_layout.addWidget(preview_group, 1)

        # == 中间面板: 实时监控 ==
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)

        # 实时位置叠加图
        monitor_group = QGroupBox("实时监控")
        monitor_layout = QVBoxLayout(monitor_group)
        self.monitor_label = QLabel()
        self.monitor_label.setMinimumSize(500, 400)
        self.monitor_label.setAlignment(Qt.AlignCenter)
        self.monitor_label.setStyleSheet("background-color: #0a0a0a; color: #666;")
        self.monitor_label.setText("等待任务开始...")
        monitor_layout.addWidget(self.monitor_label)
        mid_layout.addWidget(monitor_group, 1)

        # 进度条
        progress_group = QGroupBox("喷涂进度")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_info = QLabel("航点: 0 / 0  |  笔画: 0 / 0  |  覆盖率: 0%")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_info)
        mid_layout.addWidget(progress_group)

        # == 右侧面板: 控制 ==
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 控制区
        ctrl_group = QGroupBox("喷涂控制")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self.start_btn = QPushButton("▶  开始喷涂")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setStyleSheet("QPushButton { background-color: #2d8a2d; color: white; "
                                      "font-size: 14px; font-weight: bold; border-radius: 6px; }"
                                      "QPushButton:hover { background-color: #3aa03a; }"
                                      "QPushButton:disabled { background-color: #555; }")
        self.start_btn.clicked.connect(self._start_mission)
        self.start_btn.setEnabled(False)

        self.pause_btn = QPushButton("⏸  暂停")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setStyleSheet("QPushButton { background-color: #cc7a00; color: white; "
                                       "font-size: 13px; border-radius: 6px; }"
                                       "QPushButton:hover { background-color: #e68a00; }"
                                       "QPushButton:disabled { background-color: #555; }")
        self.pause_btn.clicked.connect(self._pause_mission)
        self.pause_btn.setEnabled(False)

        self.stop_btn = QPushButton("■  停止")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #a03030; color: white; "
                                      "font-size: 13px; font-weight: bold; border-radius: 6px; }"
                                      "QPushButton:hover { background-color: #c04040; }"
                                      "QPushButton:disabled { background-color: #555; }")
        self.stop_btn.clicked.connect(self._stop_mission)
        self.stop_btn.setEnabled(False)

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.pause_btn)
        ctrl_layout.addWidget(self.stop_btn)
        right_layout.addWidget(ctrl_group)

        # 喷涂参数
        param_group = QGroupBox("喷涂参数")
        param_layout = QFormLayout(param_group)

        self.spray_speed_spin = QDoubleSpinBox()
        self.spray_speed_spin.setRange(0.05, 1.0)
        self.spray_speed_spin.setValue(0.2)
        self.spray_speed_spin.setSingleStep(0.05)
        self.spray_speed_spin.setSuffix(" m/s")

        self.move_speed_spin = QDoubleSpinBox()
        self.move_speed_spin.setRange(0.1, 2.0)
        self.move_speed_spin.setValue(0.5)
        self.move_speed_spin.setSingleStep(0.1)
        self.move_speed_spin.setSuffix(" m/s")

        self.sampling_spin = QSpinBox()
        self.sampling_spin.setRange(1, 20)
        self.sampling_spin.setValue(5)
        self.sampling_spin.setSuffix(" pt/cm")

        self.canvas_width_spin = QDoubleSpinBox()
        self.canvas_width_spin.setRange(0.1, 10.0)
        self.canvas_width_spin.setValue(2.0)
        self.canvas_width_spin.setSuffix(" m")

        self.canvas_height_spin = QDoubleSpinBox()
        self.canvas_height_spin.setRange(0.1, 10.0)
        self.canvas_height_spin.setValue(1.5)
        self.canvas_height_spin.setSuffix(" m")

        param_layout.addRow("喷涂速度:", self.spray_speed_spin)
        param_layout.addRow("移笔速度:", self.move_speed_spin)
        param_layout.addRow("采样密度:", self.sampling_spin)
        param_layout.addRow("幕布宽度:", self.canvas_width_spin)
        param_layout.addRow("幕布高度:", self.canvas_height_spin)
        right_layout.addWidget(param_group)

        # 飞行信息
        info_group = QGroupBox("飞行信息")
        info_layout = QFormLayout(info_group)
        self.pos_x_label = QLabel("0.00 m")
        self.pos_y_label = QLabel("0.00 m")
        self.pos_z_label = QLabel("0.00 m")
        self.wp_label = QLabel("--")
        info_layout.addRow("位置 X:", self.pos_x_label)
        info_layout.addRow("位置 Y:", self.pos_y_label)
        info_layout.addRow("位置 Z:", self.pos_z_label)
        info_layout.addRow("当前航点:", self.wp_label)
        right_layout.addWidget(info_group)

        right_layout.addStretch()

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 4px;")
        right_layout.addWidget(self.status_label)

        # 组装
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(mid_panel, 3)
        main_layout.addWidget(right_panel, 2)

    def _connect_signals(self):
        self.update_odom_signal.connect(self._on_odom)
        self.update_status_signal.connect(self._on_status)
        self.update_progress_signal.connect(self._on_progress)
        self.update_coverage_signal.connect(self._on_coverage)

    # ------------------------------------------------------------------
    # 设计加载与轨迹规划
    # ------------------------------------------------------------------

    def _browse_design(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择设计图", "", "图片 (*.png *.jpg *.jpeg *.bmp *.svg)")
        if path:
            self.design_path_edit.setText(path)
            self.plan_btn.setEnabled(True)

    def _plan_trajectory(self):
        path = self.design_path_edit.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "错误", "请选择有效的设计图文件")
            return

        self._set_status("正在提取轨迹...", QColor(0, 120, 200))

        try:
            # 更新规划器参数
            self.planner = MCPPlanner(
                canvas_width_m=self.canvas_width_spin.value(),
                canvas_height_m=self.canvas_height_spin.value(),
                sampling_density=self.sampling_spin.value()
            )

            # 执行规划管线
            self.planner.load_design(path)
            self.planner.preprocess()
            self.planner.extract_trajectory()
            self.planner.optimize_path()

            # 获取参数
            spray_speed = self.spray_speed_spin.value()
            move_speed = self.move_speed_spin.value()

            # 生成航点
            self.waypoints = self.planner.generate_waypoints(
                spray_speed=spray_speed,
                move_speed=move_speed
            )
            self.total_waypoints = len(self.waypoints)
            self.total_strokes = len(self.planner.optimized_strokes)

            # 显示预览
            preview_img = self.planner.visualize_path(show=False)
            if preview_img is not None:
                self._show_preview(preview_img)

            # 发布轨迹到ROS
            traj_json = self.planner.export_as_ros_topic()
            self.trajectory_pub.publish(String(traj_json))

            self.start_btn.setEnabled(True)
            self._set_status(
                f"轨迹就绪: {self.total_strokes} 笔画, {self.total_waypoints} 航点",
                QColor(0, 180, 0)
            )

        except Exception as e:
            QMessageBox.critical(self, "规划失败", str(e))
            self._set_status(f"规划失败: {str(e)}", QColor(200, 0, 0))

    def _show_preview(self, cv_img):
        """在预览区显示OpenCV图像"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qt_img)
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # 任务控制
    # ------------------------------------------------------------------

    def _start_mission(self):
        if self.mission_completed:
            # 重置
            self.current_wp_index = 0
            self.mission_completed = False

        self.mission_active = True
        self.mission_paused = False

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸  暂停")
        self.stop_btn.setEnabled(True)
        self.plan_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)

        # 发送开始命令
        cmd = {
            'command': 'start',
            'total_waypoints': self.total_waypoints,
            'spray_speed': self.spray_speed_spin.value(),
            'move_speed': self.move_speed_spin.value()
        }
        self.cmd_pub.publish(String(json.dumps(cmd)))
        self._set_status("喷涂进行中...", QColor(0, 120, 200))

    def _pause_mission(self):
        if not self.mission_active:
            return

        self.mission_paused = not self.mission_paused
        if self.mission_paused:
            cmd = {'command': 'pause'}
            self.pause_btn.setText("▶  继续")
            self._set_status("已暂停", QColor(200, 150, 0))
        else:
            cmd = {'command': 'resume'}
            self.pause_btn.setText("⏸  暂停")
            self._set_status("喷涂进行中...", QColor(0, 120, 200))
        self.cmd_pub.publish(String(json.dumps(cmd)))

    def _stop_mission(self):
        self.mission_active = False
        self.mission_paused = False
        self.mission_completed = True

        cmd = {'command': 'stop'}
        self.cmd_pub.publish(String(json.dumps(cmd)))
        self.sprayer_pub.publish(Bool(False))

        self.start_btn.setEnabled(True)
        self.start_btn.setText("🔄  重新开始")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.plan_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)

        self._set_status("已停止", QColor(200, 0, 0))

    # ------------------------------------------------------------------
    # ROS 回调
    # ------------------------------------------------------------------

    def _ros_odom_cb(self, msg):
        self.update_odom_signal.emit(msg)

    def _ros_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            if 'current_wp' in data:
                self.update_progress_signal.emit(
                    data['current_wp'], self.total_waypoints
                )
            if 'completed' in data and data['completed']:
                self._on_mission_complete()
        except json.JSONDecodeError:
            pass

    def _ros_coverage_cb(self, msg):
        try:
            data = json.loads(msg.data)
            if 'coverage' in data:
                self.update_coverage_signal.emit(data['coverage'])
        except json.JSONDecodeError:
            pass

    def _spin_ros(self):
        try:
            rospy.get_time()
        except:
            pass

    # ------------------------------------------------------------------
    # UI 更新 (主线程)
    # ------------------------------------------------------------------

    def _on_odom(self, msg):
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        self.pos_x_label.setText(f"{self.drone_x:.2f} m")
        self.pos_y_label.setText(f"{self.drone_y:.2f} m")
        self.pos_z_label.setText(f"{self.drone_z:.2f} m")

    def _on_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color.name()}; font-weight: bold; padding: 4px;")

    def _on_progress(self, current, total):
        self.current_wp_index = current
        self.wp_label.setText(f"{current} / {total}")
        if total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)

        # 估算完成笔画
        if self.waypoints:
            strokes_completed = 0
            for i, wp in enumerate(self.waypoints):
                if i >= current:
                    break
                if wp.get('action') == 'stroke_end':
                    strokes_completed += 1
            self.completed_strokes = strokes_completed

        self.progress_info.setText(
            f"航点: {current} / {total}  |  "
            f"笔画: {self.completed_strokes} / {self.total_strokes}  |  "
            f"覆盖率: {self.coverage_percent:.1f}%"
        )

    def _on_coverage(self, pct):
        self.coverage_percent = pct

    def _on_mission_complete(self):
        self.mission_active = False
        self.mission_completed = True
        self.start_btn.setEnabled(True)
        self.start_btn.setText("🔄  新任务")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.plan_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self._set_status("喷涂完成!", QColor(0, 180, 0))
        QMessageBox.information(self, "任务完成",
                                 f"喷涂完成!\n覆盖率: {self.coverage_percent:.1f}%")

    def _update_display(self):
        """更新监控区的无人机位置叠加显示"""
        if not self.mission_active and not self.mission_completed:
            return

        # 使用规划器的可视化结果叠加当前位置
        if self.planner.design_image is not None:
            vis = self.planner.design_image.copy()

            # 绘制已完成轨迹（绿色）
            for i, wp in enumerate(self.waypoints):
                if i >= self.current_wp_index:
                    break
                if wp.get('spray'):
                    # 将物理坐标转回图像坐标
                    px = int((wp['x'] - self.planner.offset_x) / self.planner.scale_x)
                    py = int((wp['y'] - self.planner.offset_y) / self.planner.scale_y)
                    if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
                        cv2.circle(vis, (px, py), 1, (0, 255, 0), -1)

            # 绘制无人机当前位置（红色十字）
            drone_px = int((self.drone_x - self.planner.offset_x) / self.planner.scale_x)
            drone_py = int((self.drone_y - self.planner.offset_y) / self.planner.scale_y)
            if 0 <= drone_px < vis.shape[1] and 0 <= drone_py < vis.shape[0]:
                cv2.drawMarker(vis, (drone_px, drone_py), (0, 0, 255),
                               cv2.MARKER_CROSS, 12, 2)

            self._show_monitor(vis)

    def _show_monitor(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qt_img)
        scaled = pix.scaled(self.monitor_label.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
        self.monitor_label.setPixmap(scaled)

    def _set_status(self, text, color=None):
        self.status_label.setText(text)
        if color:
            self.status_label.setStyleSheet(
                f"color: {color.name()}; font-weight: bold; padding: 4px;")

    # ------------------------------------------------------------------
    # 窗口关闭
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.mission_active:
            reply = QMessageBox.question(
                self, "确认退出", "任务正在进行中，确定退出？",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                return
        # 停止喷涂
        self.sprayer_pub.publish(Bool(False))
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setPalette(app.style().standardPalette())
    station = McpPainterStation()
    station.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()