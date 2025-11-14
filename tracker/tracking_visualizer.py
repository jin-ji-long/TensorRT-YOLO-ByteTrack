import cv2
import numpy as np

_COLORS = np.array(
    [
        0.9, 0.1, 0.1,
        0.1, 0.1, 0.9,
        0.8, 0.1, 0.8,
        0.9, 0.5, 0.1,
        0.1, 0.8, 0.8,
        0.9, 0.1, 0.5,
        0.5, 0.1, 0.9,
        0.9, 0.7, 0.1,
        0.3, 0.3, 0.9,
        0.9, 0.3, 0.3,
        0.5, 0.8, 0.1,
        0.1, 0.5, 0.9,
    ]
).astype(np.float32).reshape(-1, 3)


class TrackerVisualizer:
    def __init__(self,
                 max_track_len: int = 100,
                 draw_interval: int = 5,
                 class_names: list = None):
        """跟踪可视化器：绘制目标框、轨迹和标签"""
        self.max_track_len = max_track_len  # 轨迹最大长度
        self.draw_interval = draw_interval  # 轨迹点绘制间隔
        self.class_names = class_names if class_names is not None else []
        self.track_history = {}  # {track_id: [(center_x, center_y), ...]}

    def _get_color(self, idx: int, alpha: float = 1.0):
        """根据索引获取颜色（BGR格式，支持透明度）"""
        color_rgb = _COLORS[idx % len(_COLORS)] * 255
        return (int(color_rgb[2] * alpha), int(color_rgb[1] * alpha), int(color_rgb[0] * alpha))

    def _get_text_color(self, bg_color: tuple):
        """根据背景色亮度返回黑/白文字色"""
        brightness = (0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]) / 255
        return (0, 0, 0) if brightness > 0.5 else (255, 255, 255)

    def _draw_text_with_bg(self, img, text, org, font_scale, thickness, bg_color):
        """绘制带半透明背景的文本"""
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (org[0], org[1] - text_h - baseline), (org[0] + text_w, org[1] + baseline), bg_color, -1)
        text_color = self._get_text_color(bg_color)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    def update_and_draw(self,
                        img,
                        tlwhs,
                        ids,
                        scores=None,
                        cls_ids=None,
                        frame_id=0,
                        fps=0.0):
        """更新轨迹并绘制：目标框、轨迹（带透明度衰减）、标签"""
        img = np.copy(img)
        img_h, img_w = img.shape[:2]

        # 动态参数（适配图像尺寸）
        base_scale = min(img_w, img_h) / 1000
        font_scale = max(0.4, base_scale)
        box_thickness = max(1, int(base_scale * 2))
        track_thickness = max(1, box_thickness - 1)
        text_pad = 2

        # 绘制帧信息
        info_bg_color = (20, 20, 100, 200)
        info_text = f"frame: {frame_id} | fps: {fps:.1f} | targets: {len(tlwhs)}"
        self._draw_text_with_bg(img, info_text, (10, 30), font_scale, 1, info_bg_color)

        # 绘制每个目标
        for i in range(len(tlwhs)):
            x, y, w, h = tlwhs[i]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            track_id = ids[i] if ids else -1
            score = scores[i] if scores else 0.0
            cls_id = cls_ids[i] if cls_ids else 0

            # 目标框颜色
            box_color = self._get_color(track_id if track_id != -1 else cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)

            # 绘制轨迹（仅跟踪目标）
            if track_id != -1:
                center = (int((x1 + x2) / 2), y2)
                self.track_history.setdefault(track_id, []).append(center)
                self.track_history[track_id] = self.track_history[track_id][-self.max_track_len:]
                history = self.track_history[track_id]

                # 轨迹线（间隔采样+透明度衰减）
                total_points = len(history)
                for k in range(self.draw_interval, total_points, self.draw_interval):
                    alpha = 0.3 + 0.7 * (k / total_points)
                    track_color = self._get_color(track_id, alpha=alpha)
                    cv2.line(img, history[k - self.draw_interval], history[k], track_color, track_thickness, cv2.LINE_AA)

                # 轨迹起点锚点
                if len(history) > 1:
                    cv2.circle(img, history[0], radius=track_thickness * 2, color=box_color, thickness=-1)

            # 绘制标签
            if track_id != -1:
                label = f"ID:{track_id}"
                if score > 0:
                    label += f" {score:.1%}"
            else:
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"cls{cls_id}"
                label = f"{cls_name} {score:.1%}"

            label_bg_color = (*box_color, 200)
            self._draw_text_with_bg(img, label, (x1 + text_pad, y1 + text_pad), font_scale, 1, label_bg_color)

        return img