# import os
# import torch
# import torch.nn as nn
# from tensorrt_yolo.export import torch_export  # 导入 TensorRT-YOLO 的导出函数
# import ultralytics.nn.modules.block as block  # 用于注册自定义模块

# # ========== 自定义模块定义（保持不变） ==========
# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

# def autopad(k, p=None, d=1):
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# class C3k2_SWTCC(nn.Module):
#     def __init__(self, c1, c2, shortcut=False, e=0.5):
#         super().__init__()
#         hidden = int(c2 * e)
#         self.cv1 = Conv(c1, hidden, 1, 1)
#         self.cv2 = Conv(c1, hidden, 1, 1)
#         self.wave = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
#         self.att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(hidden, hidden, 1),
#             nn.Sigmoid()
#         )
#         self.cv3 = Conv(hidden * 2, c2, 1, 1)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         y1 = self.wave(self.cv1(x))
#         y1 = y1 * self.att(y1)
#         y2 = self.cv2(x)
#         out = self.cv3(torch.cat((y1, y2), 1))
#         return x + out if self.add else out

# class SSEDown(nn.Module):
#     """Split SEAttention Downsampling"""
#     def __init__(self, c1, c2, k=3, s=2, p=1, reduction=16):
#         super().__init__()
#         self.conv = Conv(c1, c2, k, s, p)
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c2, c2 // reduction, 1, 1, 0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c2 // reduction, c2, 1, 1, 0),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.conv(x)
#         w = self.se(y)
#         return y * w

# class CFFM(nn.Module):
#     def __init__(self, c1_list, c2):
#         super().__init__()
#         c1a, c1b = c1_list
#         self.proj1 = Conv(c1a, c2, 1, 1)
#         self.proj2 = Conv(c1b, c2, 1, 1)
#         self.fuse = Conv(c2 * 2, c2, 1, 1)
#         self.att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c2, c2 // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c2 // 4, 2, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         x1, x2 = x
#         f1 = self.proj1(x1)
#         f2 = self.proj2(x2)
#         cat = torch.cat([f1, f2], dim=1)
#         fused = self.fuse(cat)

#         w = self.att(fused)  # [B,2,1,1]
#         w1 = w[:, 0:1]
#         w2 = w[:, 1:2]

#         return f1 * w1 + f2 * w2 + fused

# # ========== 注册自定义模块（保持不变） ==========
# block.C3k2_SWTCC = C3k2_SWTCC
# block.SSEDown = SSEDown
# block.CFFM = CFFM

# def export_onnx_with_tensorrt_yolo():
#     """使用 TensorRT-YOLO 仅导出 ONNX 模型"""
#     try:
#         # 配置参数（与 TensorRT-YOLO 的 torch_export 兼容）
#         pt_model_path = "D:\\Wechat\\Tencent\\xwechat_files\\wxid_tdx7j3bhzfbt22_bb03\\msg\\file\\2025-11\\best.pt"
#         output_dir = "D:\\Project\\tz_project\\byte_track_"
#         model_version = "yolo11"  # 模型版本
#         imgsz = (896, 896)     # 图像尺寸（height, width）
#         batch_size = 1           # 批次大小
#         simplify = True          # 简化 ONNX 模型

#         # 检查模型文件是否存在
#         if not os.path.exists(pt_model_path):
#             print(f"错误：模型文件不存在 - {pt_model_path}")
#             return False

#         # 调用 TensorRT-YOLO 的导出函数（内部会优先生成 ONNX）
#         print("开始使用 TensorRT-YOLO 导出 ONNX 模型...")
#         torch_export(
#             weights=pt_model_path,
#             output=output_dir,
#             version=model_version,
#             imgsz=imgsz,
#             batch=batch_size,
#             simplify=simplify
#         )

#         # 检查 ONNX 模型是否生成（仅关注 ONNX，忽略引擎）
#         onnx_name = os.path.splitext(os.path.basename(pt_model_path))[0] + ".onnx"
#         exported_onnx_path = os.path.join(output_dir, onnx_name)
#         if os.path.exists(exported_onnx_path):
#             print(f"ONNX 模型导出成功！路径：{exported_onnx_path}")
#             return True
#         else:
#             print("导出失败：未找到生成的 ONNX 模型")
#             return False

#     except Exception as e:
#         print(f"导出过程出错：{str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

# if __name__ == "__main__":
#     export_onnx_with_tensorrt_yolo()