#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import threading
import time
import os
from PIL import Image as PILImage, ImageTk, ImageDraw as PILImageDraw
import queue
import sys

# 全局变量表示是否使用ROS
USE_ROS = False

class VideoRecorderGUI:
    def __init__(self, master):
        self.master = master
        master.title("视频录制控制面板")
        
        # 初始化
        self.bridge = None
        self.is_recording = False
        self.video_writer = None
        self.frame_rate = 30
        self.resolution = (1280, 720)
        self.use_ros = False  # 默认使用本地摄像头
        self.camera_id = 0
        self.cap = None
        self.preview_active = False
        self.last_frame_time = 0  # 用于帧率控制
        self.recording_start_time = 0  # 用于记录录制开始时间
        self.recording_paused = False  # 用于暂停状态
        
        # 用于线程间通信
        self.frame_queue = queue.Queue(maxsize=1)
        self.should_exit = threading.Event()
        self.queue_lock = threading.Lock()  # 添加锁保护队列操作
        
        # 用于参数变更检测
        self._last_params = {
            'topic': '',
            'camera_id': '',
            'resolution': '',
            'framerate': '',
            'mode': ''
        }
        
        # 创建UI组件
        self.create_widgets()
        
        # 初始化ROS订阅（不立即启动）
        self.image_sub = None
        
        # 初始化预览线程（不立即启动）
        self.preview_thread = None
        self.capture_thread = None
        
        # 绑定窗口关闭事件
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # 模式选择
        ttk.Label(self.master, text="模式:").grid(row=0, column=0, padx=5, pady=5)
        self.mode_var = tk.StringVar(value="本地摄像头")  # 默认为本地摄像头
        self.mode_menu = ttk.OptionMenu(self.master, self.mode_var, "本地摄像头", "本地摄像头", "ROS", command=self.change_mode)
        self.mode_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # 相机话题选择 (初始为禁用状态)
        ttk.Label(self.master, text="相机话题:").grid(row=1, column=0, padx=5, pady=5)
        self.topic_var = tk.StringVar(value="/usb_cam/image_raw")
        self.topic_entry = ttk.Entry(self.master, textvariable=self.topic_var, state=tk.DISABLED)
        self.topic_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # 摄像头ID选择
        ttk.Label(self.master, text="摄像头ID:").grid(row=2, column=0, padx=5, pady=5)
        self.camera_id_var = tk.StringVar(value="0")
        self.camera_id_entry = ttk.Entry(self.master, textvariable=self.camera_id_var, width=10)
        self.camera_id_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # 分辨率设置
        ttk.Label(self.master, text="分辨率:").grid(row=3, column=0, padx=5, pady=5)
        self.resolution_var = tk.StringVar(value="1280x720")
        self.resolution_menu = ttk.OptionMenu(
            self.master, 
            self.resolution_var, 
            "1280x720", 
            "320x240", 
            "640x480", 
            "1280x720", 
            "1920x1080"
        )
        self.resolution_menu.grid(row=3, column=1, padx=5, pady=5)
        
        # 帧率设置
        ttk.Label(self.master, text="帧率:").grid(row=4, column=0, padx=5, pady=5)
        self.framerate_var = tk.StringVar(value="30")
        self.framerate_menu = ttk.OptionMenu(
            self.master, 
            self.framerate_var, 
            "30", 
            "15", 
            "30", 
            "60",
            "90",
            "120"
        )
        self.framerate_menu.grid(row=4, column=1, padx=5, pady=5)
        
        # 保存路径选择
        ttk.Label(self.master, text="保存路径:").grid(row=5, column=0, padx=5, pady=5)
        self.save_path_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "videos"))
        self.save_path_entry = ttk.Entry(self.master, textvariable=self.save_path_var, width=30)
        self.save_path_entry.grid(row=5, column=1, padx=5, pady=5)
        
        self.browse_btn = ttk.Button(
            self.master,
            text="浏览",
            command=self.browse_save_path
        )
        self.browse_btn.grid(row=5, column=2, padx=5, pady=5)
        
        # 预览按钮
        self.preview_btn = ttk.Button(
            self.master, 
            text="预览", 
            command=self.toggle_preview
        )
        self.preview_btn.grid(row=6, column=0, columnspan=3, pady=10)
        
        # 录制控制按钮
        self.record_btn = ttk.Button(
            self.master, 
            text="开始录制", 
            command=self.toggle_recording,
            state=tk.DISABLED
        )
        self.record_btn.grid(row=7, column=0, pady=10)
        
        self.stop_btn = ttk.Button(
            self.master, 
            text="停止录制", 
            command=self.stop_recording,
            state=tk.DISABLED
        )
        self.stop_btn.grid(row=7, column=1, columnspan=2, pady=10)
        
        # 状态标签
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(
            self.master, 
            textvariable=self.status_var
        ).grid(row=8, column=0, columnspan=3)
        
        # 检测摄像头按钮
        self.detect_btn = ttk.Button(
            self.master,
            text="检测摄像头",
            command=self.detect_cameras
        )
        self.detect_btn.grid(row=9, column=0, columnspan=3, pady=5)
    
    def init_ros(self):
        """仅在需要时初始化ROS相关功能"""
        if not self.use_ros:
            return False
            
        try:
            global rospy
            global Image
            global CvBridge
            
            # 动态导入ROS相关库
            import rospy
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            
            # 初始化ROS节点(如果尚未初始化)
            if not rospy.core.is_initialized():
                rospy.init_node('video_recorder_gui_node', anonymous=True)
                
            # 初始化bridge
            self.bridge = CvBridge()
            return True
        except ImportError as e:
            self.status_var.set(f"ROS功能不可用: {e}")
            # 自动切换到本地摄像头模式
            self.mode_var.set("本地摄像头")
            self.change_mode()
            return False
        except Exception as e:
            self.status_var.set(f"ROS初始化错误: {e}")
            return False
    
    def image_callback(self, msg):
        try:
            # 确保bridge已初始化
            if self.bridge is None:
                self.bridge = CvBridge()
                
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 使用队列传递帧，避免线程同步问题
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        except Exception as e:
            if 'rospy' in sys.modules:
                rospy.logerr(f"图像转换错误: {e}")
            else:
                print(f"图像转换错误: {e}")
            
    def get_local_camera_frame(self):
        if self.cap is None:
            try:
                self.camera_id = int(self.camera_id_var.get())
                # 对于Windows系统，使用DSHOW后端以提高性能
                if os.name == 'nt':
                    self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(self.camera_id)
                    
                # 设置摄像头缓冲区大小
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # 设置摄像头分辨率
                width, height = map(int, self.resolution_var.get().split('x'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # 设置摄像头帧率
                target_fps = float(self.framerate_var.get())
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                
                # 获取实际设置的参数
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                
                # 更新状态信息
                self.status_var.set(f"摄像头参数: {actual_width}x{actual_height} @ {actual_fps}FPS")
                
                if not self.cap.isOpened():
                    self.status_var.set(f"无法打开摄像头 {self.camera_id}")
                    self.cap = None
                    return None
            except Exception as e:
                self.status_var.set(f"摄像头ID错误: {e}")
                self.cap = None
                return None
                
        # 清空缓冲区，确保获取最新的帧
        for _ in range(2):  # 丢弃旧帧
            self.cap.grab()
            
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("无法从摄像头获取帧")
            return None
            
        return frame
    
    def capture_frames(self):
        """在独立线程中捕获帧"""
        last_capture_time = 0
        frame_interval = 1.0 / self.frame_rate
        
        while not self.should_exit.is_set():
            if self.use_ros:
                # ROS模式下不需要在此捕获帧
                time.sleep(0.01)
                continue
            
            current_time = time.time()
            elapsed = current_time - last_capture_time
            
            # 严格控制帧率
            if elapsed >= frame_interval:
                frame = self.get_local_camera_frame()
                if frame is not None:
                    # 更新队列前先清空，确保显示最新的帧
                    with self.queue_lock:
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                break
                        # 放入队列
                        self.frame_queue.put(frame)
                    last_capture_time = current_time
            
            # 计算需要等待的时间以维持精确的帧率
            wait_time = frame_interval - (time.time() - current_time)
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                # 避免CPU过载
                time.sleep(0.001)
    
    def update_preview(self):
        """处理预览和显示，在主线程中运行"""
        if not self.preview_active:
            return
            
        frame_interval = 1.0 / self.frame_rate  # 根据帧率计算帧间隔
        current_time = time.time()
            
        try:
            # 非阻塞方式获取帧
            frame = None
            try:
                # 使用超时机制避免无限等待
                frame = self.frame_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                pass
                
            if frame is not None:
                # 获取所需分辨率
                width, height = map(int, self.resolution_var.get().split('x'))
                
                # 调整大小
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # 添加录制状态信息和帧率信息
                if self.is_recording:
                    cv2.putText(frame, "RECORDING", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elapsed = time.time() - self.recording_start_time
                    cv2.putText(frame, f"{elapsed:.1f}s @{self.frame_rate}FPS", (170, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 也在预览状态显示帧率
                    cv2.putText(frame, f"Preview @{self.frame_rate}FPS", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 在主线程中显示(这段代码在主线程中执行)
                window_name = "Camera Preview"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, width, height)
                cv2.imshow(window_name, frame)
                
                # 使用waitKey同步帧率
                wait_time = max(1, int(1000 / self.frame_rate))  # 最小1ms，避免阻塞
                key = cv2.waitKey(wait_time)
                if key == 27:  # ESC键退出预览
                    self.master.after(0, self.toggle_preview)
                    return
                
                # 如果正在录制，写入帧
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
        except Exception as e:
            print(f"预览更新错误: {e}")
            
        # 计算下一帧显示的时间
        elapsed = time.time() - current_time
        next_frame_delay = max(1, int(1000 * (frame_interval - elapsed)))
        if elapsed > frame_interval:
            next_frame_delay = 1  # 如果已经超时，则尽快显示下一帧
            
        # 继续调度，使用帧率计算的延迟时间
        self.master.after(next_frame_delay, self.update_preview)

    def verify_frame_rate(self, target_fps):
        """验证摄像头支持的帧率范围"""
        if self.use_ros:
            return target_fps
            
        try:
            target_fps = float(target_fps)
            # Linux系统直接返回目标帧率，不进行验证
            if os.name != 'nt':
                return target_fps
                
            # Windows系统进行验证
            camera_id = int(self.camera_id_var.get())
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return target_fps
            
            # 设置摄像头帧率
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            return max(1.0, actual_fps)  # 确保帧率至少为1
        except Exception as e:
            self.status_var.set(f"验证帧率时出错: {e}")
            return target_fps

    def _check_params_changed(self):
        current_params = {
            'topic': self.topic_var.get(),
            'camera_id': self.camera_id_var.get(),
            'resolution': self.resolution_var.get(),
            'framerate': self.framerate_var.get(),
            'mode': self.mode_var.get()
        }
        changed = any(self._last_params[k] != current_params[k] for k in self._last_params)
        self._last_params = current_params.copy()
        return changed

    def _reset_preview(self):
        # 设置退出标志
        self.should_exit.set()
        self.preview_active = False
        
        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        
        # 释放摄像头资源
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 取消ROS订阅
        if self.image_sub:
            self.image_sub.unregister()
            self.image_sub = None

        # 主线程中关闭OpenCV窗口
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # 重置线程和事件
        self.should_exit.clear()
        
        self.status_var.set("资源已释放")

    def toggle_preview(self):
        if not self.preview_active:
            # 应用并验证参数
            try:
                # 解析分辨率
                width, height = map(int, self.resolution_var.get().split('x'))
                self.resolution = (width, height)
                
                # 应用帧率
                target_fps = float(self.framerate_var.get())
                self.frame_rate = self.verify_frame_rate(target_fps)
                
                # 摄像头ID验证
                if not self.use_ros:
                    try:
                        self.camera_id = int(self.camera_id_var.get())
                    except ValueError:
                        self.status_var.set("请输入有效的摄像头ID")
                        return
                
                # Linux下或在成功应用参数后不需要额外显示
                if os.name == 'nt':
                    self.status_var.set(f"应用参数: {width}x{height}, {self.frame_rate}FPS")
            except Exception as e:
                self.status_var.set(f"参数验证错误: {e}")
                return
            
            # 检查参数是否变更
            params_changed = self._check_params_changed()
            if params_changed:
                self._reset_preview()
            
            self.preview_active = True
            self.preview_btn.config(text="停止预览")
            
            if self.use_ros:
                # 初始化ROS
                if not self.init_ros():
                    self.status_var.set("ROS初始化失败，无法预览")
                    self.preview_active = False
                    self.preview_btn.config(text="预览")
                    return
                
                # 初始化ROS订阅
                if self.image_sub is None:
                    self.image_sub = rospy.Subscriber(
                        self.topic_var.get(), 
                        Image, 
                        self.image_callback,
                        queue_size=1
                    )
            else:
                # 启动采集线程
                self.capture_thread = threading.Thread(target=self.capture_frames)
                self.capture_thread.daemon = True
                self.capture_thread.start()
            
            # 在主线程中调度预览更新
            self.master.after(100, self.update_preview)
            
            self.status_var.set("预览中")
            self.record_btn.config(state=tk.NORMAL)
        else:
            self._reset_preview()
            self.preview_btn.config(text="预览")
            self.status_var.set("预览已停止")
            self.record_btn.config(state=tk.DISABLED)
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.pause_recording()
    
    def start_recording(self):
        # 如果是继续录制，直接恢复状态
        if self.recording_paused:
            self.is_recording = True
            self.recording_paused = False
            self.record_btn.config(text="暂停录制")
            self.status_var.set("继续录制...")
            return
            
        # 解析分辨率
        try:
            width, height = map(int, self.resolution_var.get().split('x'))
            self.resolution = (width, height)
            # 获取并验证目标帧率
            target_fps = float(self.framerate_var.get())
            self.frame_rate = self.verify_frame_rate(target_fps)
        except Exception as e:
            self.status_var.set(f"参数错误: {e}")
            return
        
        # 创建视频写入器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = self.save_path_var.get()
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"recording_{timestamp}.avi")
        
        # 使用MJPG编码器，支持更高的帧率
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        self.video_writer = cv2.VideoWriter(
            filename, 
            fourcc, 
            self.frame_rate, 
            self.resolution
        )
        
        if not self.video_writer.isOpened():
            # 如果MJPG不可用，尝试XVID
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                filename, 
                fourcc, 
                self.frame_rate, 
                self.resolution
            )
            if not self.video_writer.isOpened():
                self.status_var.set("无法初始化视频编码器")
                return
        
        self.is_recording = True
        self.recording_paused = False
        self.recording_start_time = time.time()
        self.record_btn.config(text="暂停录制")
        self.stop_btn.config(state=tk.NORMAL)
        
        codec_name = 'MJPG' if fourcc == cv2.VideoWriter_fourcc(*'MJPG') else 'XVID'
        self.status_var.set(f"开始录制: {filename} ({codec_name}, {self.frame_rate}FPS)")
    
    def pause_recording(self):
        self.is_recording = False
        self.recording_paused = True
        self.record_btn.config(text="继续录制")
        self.status_var.set("录制已暂停 - 点击继续录制按钮以继续")
    
    def stop_recording(self):
        if self.video_writer is not None:
            # 获取当前视频文件路径
            save_dir = self.save_path_var.get()
            filename = os.path.join(save_dir, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi")
            
            self.video_writer.release()
            self.video_writer = None
            
            # 显示保存信息
            self.status_var.set(f"录制已停止，视频已保存至: {filename}")
        
        self.is_recording = False
        self.recording_paused = False  # 重置暂停状态
        self.record_btn.config(text="开始录制")
        self.stop_btn.config(state=tk.DISABLED)
    
    def verify_camera_id(self):
        """验证摄像头ID是否可用"""
        try:
            camera_id = int(self.camera_id_var.get())
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                cap.release()
                self.status_var.set(f"摄像头ID {camera_id} 可用")
                return True
            else:
                self.status_var.set(f"摄像头ID {camera_id} 不可用")
                return False
        except ValueError:
            self.status_var.set("请输入有效的摄像头ID（整数）")
            return False
        except Exception as e:
            self.status_var.set(f"验证摄像头ID时出错: {e}")
            return False
    
    def detect_cameras(self):
        """检测可用的USB摄像头设备"""
        try:
            import platform
            available_ids = []
            
            if platform.system() == 'Windows':
                # 使用更可靠的设备枚举方法
                for i in range(10):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        ret, _ = cap.read()
                        if ret and cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
                            available_ids.append(str(i))
                        cap.release()
            else:
                # Linux系统使用ls /dev/video*命令检测摄像头
                try:
                    import glob
                    video_devices = glob.glob('/dev/video*')
                    
                    for device in video_devices:
                        device_id = device.split('video')[1]
                        try:
                            cap = cv2.VideoCapture(int(device_id))
                            if cap.isOpened():
                                available_ids.append(device_id)
                            cap.release()
                        except Exception as e:
                            print(f"设备 {device} 无法打开: {e}")
                except Exception as e:
                    print(f"搜索摄像头设备出错: {e}")
            
            if available_ids:
                # 更新摄像头ID输入框的默认值
                self.camera_id_var.set(available_ids[0])
                self.status_var.set(f"检测到 {len(available_ids)} 个摄像头设备")
                # 验证第一个检测到的摄像头
                self.verify_camera_id()
            else:
                self.status_var.set("未检测到摄像头设备")
                self.record_btn.config(state=tk.DISABLED)
        except Exception as e:
            self.status_var.set(f"摄像头检测错误: {e}")
            print(f"摄像头检测错误详情: {e}")
    
    def change_mode(self, *args):
        mode = self.mode_var.get()
        self.use_ros = (mode == "ROS")
        
        # 停止预览
        if self.preview_active:
            self.toggle_preview()
        
        # 重置状态
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 更新UI状态
        if self.use_ros:
            self.topic_entry.config(state=tk.NORMAL)
            self.camera_id_entry.config(state=tk.DISABLED)
            self.detect_btn.config(state=tk.DISABLED)
        else:
            self.topic_entry.config(state=tk.DISABLED)
            self.camera_id_entry.config(state=tk.NORMAL)
            self.detect_btn.config(state=tk.NORMAL)
        
        self.status_var.set(f"已切换到{mode}模式")
        self.record_btn.config(state=tk.DISABLED)

    def browse_save_path(self):
        """打开文件夹选择对话框"""
        path = filedialog.askdirectory(initialdir=self.save_path_var.get())
        if path:
            self.save_path_var.set(path)
            os.makedirs(path, exist_ok=True)
    
    def on_closing(self):
        """窗口关闭时清理资源"""
        self._reset_preview()
        if self.video_writer is not None:
            self.video_writer.release()
        self.master.destroy()

if __name__ == '__main__':
    # 检查命令行参数，允许通过 --ros 参数切换到ROS模式
    if '--ros' in sys.argv:
        USE_ROS = True
    
    root = tk.Tk()
    app = VideoRecorderGUI(root)
    
    # 如果通过命令行参数启用ROS模式
    if USE_ROS:
        app.mode_var.set("ROS")
        app.change_mode()
        # 尝试导入ROS
        try:
            import rospy
            rospy.init_node('video_recorder_gui_node')
        except ImportError:
            print("警告: 无法导入ROS库，将以本地摄像头模式运行")
            app.mode_var.set("本地摄像头")
            app.change_mode()
    
    root.mainloop()