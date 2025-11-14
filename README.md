TensorRT-YOLO + ByteTrack 目标跟踪程序
高性能 YOLO 模型部署框架，基于 TensorRT 加速

项目概述
基于 TensorRT 加速的 YOLO 目标检测器与 ByteTrack 跟踪器结合的视频目标跟踪系统。项目需要编译，编译完毕后可直接通过 Python 调用。
项目中引用于[TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO?tab=readme-ov-file)

### 1. 前置依赖
- **CUDA**：推荐版本 ≥ 11.0.1
- **TensorRT**：推荐版本 ≥ 8.6.1
- **操作系统**：Linux (x86_64 或 arm)（推荐）；Windows 亦可支持
- **额外插件**：Windows下需要额外安装VisualStudioCommunity以及CMake，linux下需要安装gcc、g++、cmake、make等工具（但一般情况下linux系统自带gcc、g++、cmake、make等工具）
请遵循以下约束：
>
> 1. 正确安装 CUDA、cuDNN、TensorRT 并配置环境变量；
> 2. 确保 cuDNN、TensorRT 版本与 CUDA 版本匹配；
> 3. 避免系统中存在多个版本的 CUDA、cuDNN、TensorRT。

### 2. 编译步骤
1. - **克隆仓库**
```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
```
2. - **安装依赖和构建**
```bash
# 安装 pybind11 用于生成 Python 绑定
pip install "pybind11[global]"
```
3. - **安装 build 工具**
```bash
pip install --upgrade build
```
4. - **配置 CMake（替换为你的实际路径**
```bash
cmake -S . -B build \
  -D TRT_PATH=/your/tensorrt/dir \
  -D BUILD_PYTHON=ON \
  -D CMAKE_INSTALL_PREFIX=/your/tensorrt-yolo/install/dir
```
5. - **编译项目**
```bash
cmake --build build -j$(nproc) --config Release --target install
```
6. - **构建 Python wheel 包**
```bash 
python -m build --wheel
```
7. - **安装生成的 wheel 包**
```bash 
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl[export]
```
执行上述指令后，tensorrt-yolo 库将被安装到指定的 CMAKE_INSTALL_PREFIX 路径中。

8. - **平台特定说明**
Linux 编译
Linux 下编译相对简单，按照上述步骤即可完成。

Windows 编译注意事项
Windows 下编译需要额外注意环境变量配置：

Path 环境变量应包含：

text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\cuDNN\v8.9.7.29\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\v8.6.1.6\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\v8.6.1.6\bin
D:\tensorrt-yolo\install\dir\bin    # 编译生成的 tensorrt_yolo.dll
D:\tensorrt-yolo\install\dir
C:\Program Files\CMake\bin

# 错误排查
常见问题 1: CUDA 工具集找不到
错误信息:

bash
CMake Error: No CUDA toolset found.
解决方案:

检查是否安装了 CUDA Visual Studio 插件

如缺失，需重新安装 CUDA 并勾选 Visual Studio 插件选项

常见问题 2: Visual Studio 版本不兼容
错误信息:

bash
fatal error C1189: #error: unsupported Microsoft Visual Studio version!
解决方案:
参考 此[解决方案](https://blog.csdn.net/lishiyu93/article/details/114599859) 处理 Visual Studio 版本兼容性问题。

### 3. 使用示例

1. 目标检测
python
```bash
from tensorrt_yolo.infer import DetectModel, InferOption
```
# 初始化模型
```bash
option = InferOption()
option.enable_swap_rb()
model = DetectModel(engine_file="your_engine_file", option=option)
```
# 执行推理
```bash
input_img = cv2.imread("test_image.jpg")
detection_result = model.predict(input_img)
```
# 处理结果
打印detection_result的结果是：
(
    num=6,
    classes=[0, 0, 0, 0, 0, 0, ],
    scores=[0.79834, 0.73584, 0.723145, 0.648926, 0.554199, 0.291016, ],
    boxes=[
        Box(left=378.846, top=227.603, right=535.507, bottom=533.044),
        Box(left=160.11, top=128.088, right=432.051, bottom=446.831),
        Box(left=348.794, top=14.7794, right=531.073, bottom=318.496),
        Box(left=122.176, top=332.044, right=412.838, bottom=536),
        Box(left=1.10844, top=230.682, right=193.118, bottom=528.118),
        Box(left=127.596, top=2.21691, right=403.971, bottom=279.577),
    ]
)
num为检测到的数量，classes为检测的类别，scores为检测的置信度，boxes为检测的框，left 对应 x1、top 对应 y1、right 对应 x2、bottom 对应 y21。
2. 目标跟踪
python
```bash
from tracker.byte_tracker import BYTETracker
```
# 初始化跟踪器
```bash
tracker = BYTETracker(
    track_thresh=0.5, 
    track_buffer=30, 
    match_thresh=0.78,
    fuse_score=True,
    frame_rate=30
)
```
# 更新跟踪器（使用检测结果）
# dets_np 格式: [x1, y1, x2, y2, score]
# img_info 为图片宽高信息
track_outputs = tracker.update(dets_np, img_info, img_info)
多线程使用
python
# 如需多线程，对模型进行克隆
model_clone = model.clone()


### 4. 模型转换
1. - **导出 ONNX 模型**
```bash
trtyolo export \
  -w model.pt \
  -v yolovx \
  -o output.onnx \
  -b 1 \
  --imgsz 1088,1920 \
  -s
```
参数说明:

-w: 原始 PyTorch 模型路径

-v: YOLO 模型版本

-o: ONNX 模型输出路径

-b: 批次大小

--imgsz: 模型输入尺寸

-s: 简化模型

2. - **转换为 TensorRT 引擎**
```bash
trtexec \
  --onnx=output.onnx \
  --saveEngine=output.engine \
  --fp16
```
参数说明:

--onnx: ONNX 模型路径

--saveEngine: TensorRT 引擎输出路径

--fp16: 使用 FP16 精度

输出格式说明
检测结果 detection_result 包含以下字段:

num: 检测到的目标数量

classes: 类别 ID 列表

scores: 置信度分数列表

boxes: 边界框列表，每个框包含:

left: x1 坐标

top: y1 坐标

right: x2 坐标

bottom: y2 坐标

目录结构
text
byte_track_tracker/
├── tracker/           # 跟踪器实现
│   ├── byte_tracker.py
│   └── __init__.py
├── demo.py           # 使用示例
└── README.md         # 说明文档
按照上述步骤，您可以成功编译并使用 TensorRT-YOLO + ByteTrack 目标跟踪系统。







