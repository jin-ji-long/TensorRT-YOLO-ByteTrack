import cv2
import numpy as np
from tensorrt_yolo.infer import DetectModel, InferOption
from tracker.byte_tracker import BYTETracker


class YOLODetector:
    """YOLO检测器（支持模型克隆，用于多线程）"""
    def __init__(self, engine_path, conf_threshold=0.5, target_class_id=0):
        """
        初始化检测器
        :param engine_path: TensorRT引擎文件路径
        :param conf_threshold: 置信度阈值
        :param target_class_id: 目标类别ID（如0表示person）
        """
        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        self.model = self._load_model(engine_path)

    def _load_model(self, engine_path):
        """加载基础模型"""
        option = InferOption()
        option.enable_swap_rb()  # 转换BGR->RGB
        return DetectModel(engine_file=engine_path, option=option)

    def clone(self):
        """克隆模型实例（关键：用于多线程，避免资源竞争）"""
        # 创建新实例，克隆底层模型（共享引擎，独立上下文）
        cloned = YOLODetector(
            engine_path=None,  # 无需重复传入路径
            conf_threshold=self.conf_threshold,
            target_class_id=self.target_class_id
        )
        cloned.model = self.model.clone()  # 核心克隆操作
        return cloned

    def detect(self, frame):
        """
        检测单帧图像
        :param frame: 输入帧（BGR格式）
        :return: 检测框(xyxy)和分数，格式：(np.array([[x1,y1,x2,y2], ...]), np.array([score1, ...]))
        """
        result = self.model.predict(frame)
        boxes, scores = [], []
        
        for i in range(result.num):
            # 筛选目标类别和置信度
            if result.classes[i] == self.target_class_id and result.scores[i] >= self.conf_threshold:
                box = result.boxes[i]
                boxes.append([box.left, box.top, box.right, box.bottom])  # xyxy格式
                scores.append(result.scores[i])
        
        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)


class SimpleTracker:
    """追踪器封装（每个线程独立实例）"""
    def __init__(self, track_thresh=0.5, track_buffer=30, frame_rate=30):
        """初始化追踪器"""
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=0.8,
            fuse_score=True,
            frame_rate=frame_rate
        )

    def update(self, det_boxes, det_scores, img_h, img_w):
        """
        更新追踪结果
        :param det_boxes: 检测框(xyxy格式，np.array)
        :param det_scores: 检测分数(np.array)
        :param img_h: 图像高度
        :param img_w: 图像宽度
        :return: 追踪框(tlwh格式)和追踪ID，格式：([(x,y,w,h), ...], [id1, id2, ...])
        """
        # 格式化检测结果为[N,5]：(x1,y1,x2,y2,score)
        dets = np.hstack((det_boxes, det_scores.reshape(-1, 1))) if len(det_boxes) > 0 else np.empty((0, 5))
        # 更新追踪器
        online_targets = self.tracker.update(dets, [img_h, img_w], [img_h, img_w])
        
        # 提取结果：tlwh格式(左上角x,y + 宽高)和追踪ID
        track_boxes, track_ids = [], []
        for target in online_targets:
            track_boxes.append(target.tlwh)
            track_ids.append(target.track_id)
        
        return track_boxes, track_ids


# --------------------------
# 示例：如何创建和使用
# --------------------------
if __name__ == "__main__":
    # 1. 配置参数
    ENGINE_PATH = "yolov8n.engine"  # 替换为你的模型路径
    TARGET_CLASS_ID = 0  # 检测目标：0=person（根据模型类别调整）
    CONF_THRESH = 0.5    # 检测置信度阈值

    # 2. 创建基础检测器（主线程中初始化）
    base_detector = YOLODetector(
        engine_path=ENGINE_PATH,
        conf_threshold=CONF_THRESH,
        target_class_id=TARGET_CLASS_ID
    )

    # 3. 克隆检测器（用于多线程，每个线程一个独立实例）
    # （单线程可直接用base_detector，多线程必须克隆）
    detector_clone1 = base_detector.clone()  # 线程1用
    detector_clone2 = base_detector.clone()  # 线程2用

    # 4. 创建追踪器（每个线程必须独立创建，不能共享）
    tracker1 = SimpleTracker(track_thresh=0.5, track_buffer=30)  # 线程1用
    tracker2 = SimpleTracker(track_thresh=0.5, track_buffer=30)  # 线程2用

    # 5. 单帧处理示例（模拟单线程）
    print("=== 单帧处理示例 ===")
    frame = cv2.imread("test.jpg")  # 替换为你的图像
    if frame is not None:
        # 检测
        det_boxes, det_scores = base_detector.detect(frame)
        print(f"检测到 {len(det_boxes)} 个目标")
        
        # 追踪
        h, w = frame.shape[:2]
        track_boxes, track_ids = tracker1.update(det_boxes, det_scores, h, w)
        print(f"追踪到 {len(track_ids)} 个目标，ID: {track_ids}")

    # 6. 多线程使用提示（核心逻辑）
    print("\n=== 多线程使用提示 ===")
    print("""
    # 线程函数示例
    def thread_func(detector, tracker, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            # 检测（用线程专属的detector）
            det_boxes, det_scores = detector.detect(frame)
            
            # 追踪（用线程专属的tracker）
            h, w = frame.shape[:2]
            track_boxes, track_ids = tracker.update(det_boxes, det_scores, h, w)
            
            # 处理结果...
            current_frame += 1

    # 启动线程
    import threading
    threading.Thread(target=thread_func, args=(detector_clone1, tracker1, "video.mp4", 0, 100)).start()
    threading.Thread(target=thread_func, args=(detector_clone2, tracker2, "video.mp4", 101, 200)).start()
    """)