#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP (Motion Control Planning) Trajectory Planner
核心功能：AI生图 -> 轮廓提取 -> 轨迹采样 -> MCP路径规划

Workflow:
  1. load_design()     - 加载AI生成的设计图
  2. extract_trajectory() - Canny边缘检测 + findContours笔画提取
  3. optimize_path()   - TSP优化笔画顺序
  4. generate_waypoints() - 图像坐标映射到无人机物理坐标
  5. smooth_trajectory()  - 样条插值平滑
"""

import cv2
import numpy as np
import math
import yaml
import os
from scipy import interpolate


class MCPPlanner:
    """MCP轨迹规划器 - 将AI设计图转换为无人机喷涂轨迹"""

    def __init__(self, canvas_width_m=2.0, canvas_height_m=1.5,
                 flight_height=1.22, sampling_density=5):
        self.canvas_width_m = canvas_width_m      # 幕布物理宽度(米)
        self.canvas_height_m = canvas_height_m     # 幕布物理高度(米)
        self.flight_height = flight_height         # 喷涂飞行高度(米)
        self.sampling_density = sampling_density   # 采样密度(点/厘米)

        # 图像相关
        self.design_image = None        # 原始设计图
        self.binary_image = None        # 二值化图像
        self.image_width = 0
        self.image_height = 0

        # 轨迹数据
        self.contours = []              # OpenCV轮廓列表
        self.strokes = []               # 笔画列表 [{id, points, bbox}]
        self.optimized_strokes = []     # TSP优化后的笔画顺序
        self.waypoints = []             # 最终无人机航点

        # 映射参数
        self.scale_x = 1.0              # 像素->米 X方向缩放
        self.scale_y = 1.0              # 像素->米 Y方向缩放
        self.offset_x = 0.0
        self.offset_y = 0.0

    # ------------------------------------------------------------------
    # 1. 设计加载
    # ------------------------------------------------------------------

    def load_design(self, image_path):
        """加载AI生成的设计图"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"设计文件不存在: {image_path}")

        self.design_image = cv2.imread(image_path)
        if self.design_image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        self.image_height, self.image_width = self.design_image.shape[:2]

        # 计算像素到物理坐标的映射
        self.scale_x = self.canvas_width_m / self.image_width
        self.scale_y = self.canvas_height_m / self.image_height

        print(f"[MCP] 加载设计图: {image_path}")
        print(f"      尺寸: {self.image_width}x{self.image_height} px")
        print(f"      映射: {self.scale_x*100:.2f} cm/px (X), {self.scale_y*100:.2f} cm/px (Y)")
        return True

    # ------------------------------------------------------------------
    # 2. 轮廓/轨迹提取
    # ------------------------------------------------------------------

    def preprocess(self, invert=True, blur_ksize=5, canny_low=50, canny_high=150):
        """图像预处理：灰度化 -> 高斯模糊 -> Canny边缘检测"""
        gray = cv2.cvtColor(self.design_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)

        if invert:
            self.binary_image = cv2.bitwise_not(edges)
        else:
            self.binary_image = edges

        return self.binary_image

    def extract_trajectory(self, min_contour_area=20, max_contour_area=None):
        """从预处理的二值图像中提取轮廓轨迹"""
        if self.binary_image is None:
            self.preprocess()

        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            cv2.bitwise_not(self.binary_image) if not np.any(self.binary_image > 0)
            else self.binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            print("[MCP] 未检测到任何轮廓")
            return []

        if max_contour_area is None:
            max_contour_area = self.image_width * self.image_height * 0.8

        # 过滤轮廓并构建笔画
        self.strokes = []
        self.contours = []
        stroke_id = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area or area > max_contour_area:
                continue

            # 轮廓近似以降低点数
            epsilon = 0.5
            approx = cv2.approxPolyDP(cnt, epsilon, False)

            # 沿轮廓等距采样
            points = self._sample_contour(approx, self.sampling_density)

            if len(points) < 3:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            self.contours.append(approx)
            self.strokes.append({
                'id': stroke_id,
                'points': points,          # [(x_px, y_px), ...]
                'bbox': (x, y, w, h),
                'area': area,
                'length': cv2.arcLength(cnt, False)
            })
            stroke_id += 1

        # 按面积降序排列（先画大面积主体）
        self.strokes.sort(key=lambda s: s['area'], reverse=True)
        for i, s in enumerate(self.strokes):
            s['id'] = i

        print(f"[MCP] 提取到 {len(self.strokes)} 个笔画")
        return self.strokes

    def _sample_contour(self, contour, density):
        """沿轮廓等距采样"""
        perimeter = cv2.arcLength(contour, False)
        num_points = max(3, int(perimeter * density / 10))  # density=点/厘米
        # 如果轮廓已经用CHAIN_APPROX_NONE提取则点数已足够
        if len(contour) <= num_points:
            return [(pt[0][0], pt[0][1]) for pt in contour]

        # 等距重新采样
        sampled = []
        total_length = perimeter
        step = total_length / num_points
        current = 0.0

        # 展平轮廓点
        pts = [(pt[0][0], pt[0][1]) for pt in contour]

        for i in range(len(pts)):
            if current >= len(sampled) * step:
                sampled.append(pts[i])
            if i < len(pts) - 1:
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                seg_len = math.sqrt(dx*dx + dy*dy)
                current += seg_len

        return sampled

    # ------------------------------------------------------------------
    # 3. 路径优化 (TSP)
    # ------------------------------------------------------------------

    def optimize_path(self):
        """TSP优化笔画顺序，最小化空驶距离"""
        if len(self.strokes) <= 1:
            self.optimized_strokes = self.strokes[:]
            return self.optimized_strokes

        n = len(self.strokes)
        visited = [False] * n
        order = []
        current = 0  # 从最大面积笔画开始
        visited[current] = True
        order.append(current)

        while len(order) < n:
            best_dist = float('inf')
            best_idx = -1
            # 找当前笔画终点到未访问笔画起点的最近距离
            cx, cy = self._stroke_end(self.strokes[current])
            for i in range(n):
                if visited[i]:
                    continue
                sx, sy = self.strokes[i]['points'][0]
                dx = sx - cx
                dy = sy - cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            visited[best_idx] = True
            order.append(best_idx)
            current = best_idx

        self.optimized_strokes = [self.strokes[i] for i in order]

        # 优化前后翻转判断：对每个笔画，如果终点离下一笔画起点更近则翻转
        for i in range(len(self.optimized_strokes) - 1):
            pts = self.optimized_strokes[i]['points']
            next_pts = self.optimized_strokes[i+1]['points']
            dist_start = math.sqrt((pts[-1][0] - next_pts[0][0])**2 +
                                    (pts[-1][1] - next_pts[0][1])**2)
            dist_end = math.sqrt((pts[0][0] - next_pts[0][0])**2 +
                                  (pts[0][1] - next_pts[0][1])**2)
            if dist_end < dist_start:
                self.optimized_strokes[i]['points'] = pts[::-1]

        print(f"[MCP] TSP路径优化完成")
        return self.optimized_strokes

    def _stroke_end(self, stroke):
        """获取笔画终点"""
        return stroke['points'][-1]

    # ------------------------------------------------------------------
    # 4. 轨迹平滑 (样条插值)
    # ------------------------------------------------------------------

    def smooth_trajectory(self, stroke_points, smoothing=0.0):
        """对轨迹进行三次样条插值平滑"""
        pts = np.array(stroke_points)
        if len(pts) < 4:
            return stroke_points

        x = pts[:, 0]
        y = pts[:, 1]

        # 参数化
        t = np.arange(len(pts))
        tt = np.linspace(0, len(pts)-1, len(pts)*2)

        try:
            sx = interpolate.UnivariateSpline(t, x, s=smoothing)
            sy = interpolate.UnivariateSpline(t, y, s=smoothing)
            x_smooth = sx(tt)
            y_smooth = sy(tt)
            return list(zip(x_smooth.tolist(), y_smooth.tolist()))
        except Exception as e:
            print(f"[MCP] 平滑失败，使用原始点: {e}")
            return stroke_points

    # ------------------------------------------------------------------
    # 5. 航点生成 (图像坐标 -> 无人机物理坐标)
    # ------------------------------------------------------------------

    def generate_waypoints(self, move_speed=0.5, spray_speed=0.2,
                           smooth=True, smoothing_param=0.0):
        """生成最终无人机航点列表

        将图像坐标映射到无人机物理坐标系，生成包含喷涂控制信号的航点。
        """
        if not self.optimized_strokes:
            self.optimize_path()

        self.waypoints = []
        waypoint_id = 0

        for i, stroke in enumerate(self.optimized_strokes):
            points = stroke['points']

            if smooth:
                points = self.smooth_trajectory(points, smoothing_param)

            # 移笔移动（空驶到笔画起点）—— 关闭喷涂
            if i > 0:
                sx, sy = points[0]
                wx = sx * self.scale_x + self.offset_x
                wy = sy * self.scale_y + self.offset_y
                self.waypoints.append({
                    'id': waypoint_id,
                    'x': round(wx, 4),
                    'y': round(wy, 4),
                    'z': self.flight_height,
                    'spray': False,
                    'speed': move_speed,
                    'stroke_id': stroke['id'],
                    'action': 'move_to_stroke'
                })
                waypoint_id += 1

            # 喷涂轨迹点
            for j, (px, py) in enumerate(points):
                wx = px * self.scale_x + self.offset_x
                wy = py * self.scale_y + self.offset_y
                self.waypoints.append({
                    'id': waypoint_id,
                    'x': round(wx, 4),
                    'y': round(wy, 4),
                    'z': self.flight_height,
                    'spray': True,
                    'speed': spray_speed,
                    'stroke_id': stroke['id'],
                    'action': 'spray'
                })
                waypoint_id += 1

            # 笔画结束 - 关闭喷涂
            if i < len(self.optimized_strokes) - 1:
                ex, ey = points[-1]
                wx = ex * self.scale_x + self.offset_x
                wy = ey * self.scale_y + self.offset_y
                self.waypoints.append({
                    'id': waypoint_id,
                    'x': round(wx, 4),
                    'y': round(wy, 4),
                    'z': self.flight_height,
                    'spray': False,
                    'speed': move_speed,
                    'stroke_id': stroke['id'],
                    'action': 'stroke_end'
                })
                waypoint_id += 1

        total_spray = sum(1 for wp in self.waypoints if wp['spray'])
        total_move = sum(1 for wp in self.waypoints if not wp['spray'])
        print(f"[MCP] 航点生成完成: {len(self.waypoints)} 个")
        print(f"      喷涂点: {total_spray}, 移笔点: {total_move}")
        return self.waypoints

    def set_origin_offset(self, origin_x, origin_y):
        """设置幕布原点在无人机坐标系中的偏移"""
        self.offset_x = origin_x
        self.offset_y = origin_y

    # ------------------------------------------------------------------
    # 6. 可视化
    # ------------------------------------------------------------------

    def visualize_path(self, title="MCP Trajectory Preview", show=True):
        """可视化规划好的轨迹路径"""
        if self.design_image is None:
            print("[MCP] 请先加载设计图")
            return None

        vis = self.design_image.copy()

        # 绘制轨迹
        for i, stroke in enumerate(self.optimized_strokes):
            pts = stroke['points']
            color = self._stroke_color(i)
            for j in range(len(pts) - 1):
                cv2.line(vis,
                         (int(pts[j][0]), int(pts[j][1])),
                         (int(pts[j+1][0]), int(pts[j+1][1])),
                         color, 1, cv2.LINE_AA)

            # 标注笔画编号
            if pts:
                cv2.putText(vis, str(i),
                            (int(pts[0][0]), int(pts[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 添加统计信息
        info = [
            f"Strokes: {len(self.optimized_strokes)}",
            f"Waypoints: {len(self.waypoints)}",
            f"Canvas: {self.canvas_width_m:.1f}x{self.canvas_height_m:.1f}m"
        ]
        for i, text in enumerate(info):
            cv2.putText(vis, text, (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if show:
            cv2.imshow(title, vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis

    def _stroke_color(self, idx):
        """根据笔画索引生成不同颜色"""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
            (128, 0, 255)
        ]
        return colors[idx % len(colors)]

    # ------------------------------------------------------------------
    # 7. 导出
    # ------------------------------------------------------------------

    def export_waypoints(self, filepath):
        """导出航点到YAML文件"""
        if not self.waypoints:
            self.generate_waypoints()

        data = {
            'metadata': {
                'design_source': 'ai_generated',
                'canvas_width_m': self.canvas_width_m,
                'canvas_height_m': self.canvas_height_m,
                'flight_height_m': self.flight_height,
                'sampling_density': self.sampling_density,
                'total_strokes': len(self.optimized_strokes),
                'total_waypoints': len(self.waypoints)
            },
            'waypoints': self.waypoints
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        print(f"[MCP] 航点已导出: {filepath}")
        return filepath

    def export_as_ros_topic(self):
        """将航点转换为ROS消息格式(JSON字符串)"""
        import json
        return json.dumps({
            'waypoints': self.waypoints,
            'metadata': {
                'total': len(self.waypoints),
                'strokes': len(self.optimized_strokes)
            }
        }, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 完整管线
    # ------------------------------------------------------------------

    def run_pipeline(self, image_path, origin_x=0, origin_y=0):
        """执行完整的 MCP 规划管线

        Args:
            image_path: AI设计图路径
            origin_x: 幕布原点X偏移(无人机坐标系)
            origin_y: 幕布原点Y偏移(无人机坐标系)
        """
        print("=" * 50)
        print("MCP Trajectory Planning Pipeline")
        print("=" * 50)

        # 1. 加载
        self.load_design(image_path)
        self.set_origin_offset(origin_x, origin_y)

        # 2. 预处理
        self.preprocess()

        # 3. 轨迹提取
        self.extract_trajectory()

        # 4. 路径优化
        self.optimize_path()

        # 5. 航点生成
        self.generate_waypoints()

        # 6. 可视化
        self.visualize_path()

        print("=" * 50)
        print("Pipeline Complete")
        print("=" * 50)
        return self.waypoints


if __name__ == '__main__':
    # 测试
    import sys
    if len(sys.argv) > 1:
        planner = MCPPlanner()
        planner.run_pipeline(sys.argv[1])
    else:
        print("用法: python mcp_planner.py <design_image.png>")