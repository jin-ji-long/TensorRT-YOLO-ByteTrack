# config.py
"""配置参数文件，替代原yaml配置"""

# 检测器配置
DETECTOR_CONFIG = {
    "engine_path": "best.engine",
    "target_class": "person",
    "conf_threshold": 0.4,
    "class_labels": ["person"]
}

# 跟踪器配置
TRACKER_CONFIG = {
    "track_thresh": 0.4,
    "track_buffer": 100,
    "match_thresh": 0.78,
    "fuse_score": True,
    "frame_rate": 25
}

# 可视化配置
VISUALIZER_CONFIG = {
    "max_track_len": 100,
    "draw_interval": 5,
    "scale": 0.8
}

# 视频配置
VIDEO_CONFIG = {
    "path": "D:\\Wechat\\Tencent\\xwechat_files\\wxid_tdx7j3bhzfbt22_bb03\\msg\\file\\2025-11\\video\\video\\output.mp4",
    "output_path": "output_tracking_result.mp4"
}