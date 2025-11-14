import cv2
import numpy as np
import time

# 从config.py导入配置参数
from config import DETECTOR_CONFIG, TRACKER_CONFIG, VISUALIZER_CONFIG, VIDEO_CONFIG

from tensorrt_yolo.infer import DetectModel, InferOption
from tracker.byte_tracker import BYTETracker
from tracker.tracking_visualizer import TrackerVisualizer


class YOLOTracker:
    def __init__(self):
        """初始化YOLO跟踪器（直接从config.py读取配置）"""
        self.load_config()
        self.setup_components()
        
        # 统计信息
        self.frame_count = 0
        self.total_inference_time = 0
        self.detection_times = []
        self.tracking_times = []
        
    def load_config(self):
        """加载配置（从config.py导入）"""
        self.detector_cfg = DETECTOR_CONFIG
        self.tracker_cfg = TRACKER_CONFIG
        self.visualizer_cfg = VISUALIZER_CONFIG
        self.video_cfg = VIDEO_CONFIG
        
        # 目标类别ID
        self.target_class_id = self.detector_cfg['class_labels'].index(
            self.detector_cfg['target_class']
        ) if self.detector_cfg['target_class'] in self.detector_cfg['class_labels'] else -1
        
        if self.target_class_id == -1:
            raise ValueError(f"目标类别 {self.detector_cfg['target_class']} 不在类别映射中")
    
    def setup_components(self):
        """初始化各个组件"""
        # 初始化检测器
        option = InferOption()
        option.enable_swap_rb()  # BGR转RGB
        self.model = DetectModel(
            engine_file=self.detector_cfg['engine_path'], 
            option=option
        )
        
        # 初始化跟踪器
        self.tracker = BYTETracker(
            track_thresh=self.tracker_cfg['track_thresh'],
            track_buffer=self.tracker_cfg['track_buffer'],
            match_thresh=self.tracker_cfg['match_thresh'],
            fuse_score=self.tracker_cfg['fuse_score'],
            frame_rate=self.tracker_cfg['frame_rate']
        )
        
        # 初始化可视化器（保留）
        self.visualizer = TrackerVisualizer(
            max_track_len=self.visualizer_cfg['max_track_len'],
            draw_interval=self.visualizer_cfg['draw_interval']
        )
        
        # 打开视频文件
        self.cap = cv2.VideoCapture(self.video_cfg['path'])
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.video_cfg['path']}")
        
        # 获取视频信息
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 视频写入器初始化为None
        self.video_writer = None
    
    def setup_video_writer(self, frame):
        """初始化视频写入器"""
        # 计算输出尺寸
        vis_h, vis_w = frame.shape[:2]
        new_w = int(vis_w * self.visualizer_cfg['scale'])
        new_h = int(vis_h * self.visualizer_cfg['scale'])
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_cfg['output_path'],
            fourcc,
            self.original_fps,
            (new_w, new_h)
        )
    
    def process_detection(self, frame):
        """处理目标检测，返回x1,y1,x2,y2格式的检测框"""
        det_boxes = []  # 存储格式：(x1, y1, x2, y2)
        det_cls_ids = []
        det_scores = []
        
        # 执行检测
        result = self.model.predict(frame)
        
        # 筛选有效目标
        for i in range(result.num):
            cls_id = result.classes[i]
            score = result.scores[i]
            if (cls_id == self.target_class_id) and (score >= self.detector_cfg['conf_threshold']):
                box = result.boxes[i]
                x1, y1 = box.left, box.top  # 左上角坐标
                x2, y2 = box.right, box.bottom  # 右下角坐标
                det_boxes.append((x1, y1, x2, y2))
                det_cls_ids.append(cls_id)
                det_scores.append(score)
        
        return det_boxes, det_cls_ids, det_scores
    
    def process_tracking(self, det_boxes, det_scores, img_info, frame_id):
        """处理目标跟踪，使用x1,y1,x2,y2格式构建跟踪器输入"""
        # 构建跟踪器输入格式：[x1, y1, x2, y2, score]
        dets_np = np.array(
            [[x1, y1, x2, y2, s] for (x1, y1, x2, y2), s in zip(det_boxes, det_scores)],
            dtype=np.float32
        ) if det_boxes else np.empty((0, 5), dtype=np.float32)
        
        # 更新跟踪器
        online_targets = self.tracker.update(
            dets_np, img_info, img_info
        )
        
        # 提取跟踪结果
        track_tlwhs = []
        track_ids = []
        track_scores = []
        
        for target in online_targets:
            x1, y1, w, h = target.tlwh  # tlwh格式：(左上角x, 左上角y, 宽, 高)
            track_tlwhs.append((x1, y1, w, h))
            track_ids.append(target.track_id)
            track_scores.append(target.score)
        
        return track_tlwhs, track_ids, track_scores
    
    def calculate_fps(self, start_time):
        """计算实时FPS"""
        elapsed = time.time() - start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def process_frame(self, frame):
        """处理单帧图像（移除推理信息绘制）"""
        self.frame_count += 1
        
        # 1. 目标检测
        detection_start = time.time()
        det_boxes, det_cls_ids, det_scores = self.process_detection(frame)
        detection_time = time.time() - detection_start
        
        # 2. 目标跟踪
        tracking_start = time.time()
        img_info = [frame.shape[0], frame.shape[1]]  # 原始图像高、宽
        track_tlwhs, track_ids, track_scores = self.process_tracking(
            det_boxes, det_scores, img_info, self.frame_count
        )
        tracking_time = time.time() - tracking_start
        
        # 更新统计信息
        self.total_inference_time += detection_time + tracking_time
        self.detection_times.append(detection_time)
        self.tracking_times.append(tracking_time)
        
        # 3. 可视化（仅保留TrackerVisualizer的绘制）
        fps = self.calculate_fps(self.start_time)
        vis_frame = self.visualizer.update_and_draw(
            img=frame,
            tlwhs=track_tlwhs,
            ids=track_ids,
            scores=track_scores,
            cls_ids=det_cls_ids[:len(track_tlwhs)],
            frame_id=self.frame_count,
            fps=fps
        )
        
        return vis_frame if vis_frame is not None else frame
    
    def run(self, save_video=False, show_video=True):
        """运行主循环（移除show_inference_time参数）"""
        print("开始处理视频...")
        self.start_time = time.time()
        
        # 初始化视频写入器（如果需要保存视频）
        if save_video:
            # 读取第一帧获取尺寸
            ret, first_frame = self.cap.read()
            if not ret:
                raise RuntimeError("无法读取视频第一帧")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
            
            # 处理第一帧以获取可视化尺寸
            processed_first_frame = self.process_frame(first_frame)
            if processed_first_frame is None:
                processed_first_frame = first_frame
            
            # 设置视频写入器
            self.setup_video_writer(processed_first_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 再次重置到第一帧
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 处理当前帧
            processed_frame = self.process_frame(frame)
            if processed_frame is None:
                continue
            
            # 缩放图像
            vis_h, vis_w = processed_frame.shape[:2]
            new_w = int(vis_w * self.visualizer_cfg['scale'])
            new_h = int(vis_h * self.visualizer_cfg['scale'])
            resized_frame = cv2.resize(
                processed_frame, (new_w, new_h), 
                interpolation=cv2.INTER_AREA
            )
            
            # 保存视频
            if save_video and self.video_writer is not None:
                self.video_writer.write(resized_frame)
            
            # 显示视频
            if show_video:
                cv2.imshow("TensorRT-YOLO + ByteTrack", resized_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # 清理资源
        self.cleanup(save_video)
        
    def cleanup(self, save_video):
        """清理资源"""
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        self.print_statistics(save_video)
    
    def print_statistics(self, save_video):
        """打印运行统计信息"""
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        print("\n=== 运行统计 ===")
        print(f"总帧数: {self.frame_count}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        print(f"平均检测时间: {np.mean(self.detection_times)*1000:.2f}ms")
        print(f"平均跟踪时间: {np.mean(self.tracking_times)*1000:.2f}ms")
        print(f"总推理时间: {self.total_inference_time:.2f}秒")
        
        if save_video:
            print(f"输出视频已保存到: {self.video_cfg['output_path']}")


def main():
    """主函数"""
    try:
        # 设置开关（移除show_inference_time）
        save_video = True      # 是否保存结果视频
        show_video = True      # 是否显示视频窗口
        
        # 初始化跟踪器
        tracker = YOLOTracker()
        
        # 运行跟踪
        tracker.run(
            save_video=save_video,
            show_video=show_video
        )
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()