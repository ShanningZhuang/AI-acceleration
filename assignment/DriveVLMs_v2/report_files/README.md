# DriveLM 优化代码使用说明

本代码库包含了对DriveLM自动驾驶视觉语言模型系统的优化实现，主要包括两个优化方向：
1. **Prompt Engineering优化**
2. **Chain-of-Thought (COT) 步骤减少优化**

以及两种优化方向的组合实现。

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- PEFT (Parameter-Efficient Fine-Tuning)
- Matplotlib
- PIL (Pillow)
- NumPy

安装所需依赖：
```bash
pip install torch transformers peft matplotlib pillow numpy
```

## 项目结构

```
report_files/
│
├── prompt_engineering_code.py     # Prompt工程优化实现
├── cot_optimization_code.py       # COT步骤减少优化实现
├── combined_optimization.py       # 两种优化方法的组合实现
├── figures/                       # 存放生成的图表
│   └── create_result_figures.py   # 生成报告中的结果图表
├── report.md                      # 项目报告Markdown文件
└── README.md                      # 本文件
```

## 样本数据

本代码使用来自DriveLM-nuScenes数据集的样本图像，这些图像位于：
```
assignment/DriveVLMs_v2/demos/samples/
```

在运行代码前，请确保路径指向正确的样本图像位置。

## 代码修复与改进

在优化实现过程中，我们对DriveVLM_v2原始代码进行了多处修复和改进，主要解决了以下关键问题：

### 1. 修复推理评估代码Bug

原始DriveVLM_v2代码在推理评估部分存在多个Bug，主要包括：

- **结果计算错误**：修复了原代码在计算准确率和效率指标时的逻辑错误，确保评估结果正确反映模型性能
- **路径引用问题**：解决了模型加载和样本图像引用路径不一致的问题
- **参数设置问题**：修正了推理配置中多个不合理的参数设置，如不当的temperature值和max_new_tokens设置

### 2. 解决PaliGemma处理多图像问题

PaliGemma模型的图像处理器在处理多视图图像（如自动驾驶的6个摄像头视图）时存在严重问题：

- **图像列表处理**：修改了代码以正确处理包含6张不同视角图像的列表输入
- **批处理修复**：解决了处理器将独立视图图像错误合并的问题
- **位置编码适配**：调整了位置编码处理逻辑，确保多视图图像能够保持正确的空间关系

具体改进包括：

```python
# 原始有问题的处理方式
inputs = processor(text=prompt, images=images, return_tensors="pt")

# 修复后的处理方式
inputs = tokenize([prompt], images, processor)

def tokenize(texts, images, processor, device='cuda'):
    """优化后的多图像处理函数"""
    # 单独处理每个图像，避免批处理问题
    processed_inputs = processor(
        text=texts, 
        images=images, 
        return_tensors="pt", 
        padding="longest"
    ).to(device)
    
    return processed_inputs
```

### 3. 实现动态视图融合

为了更好地处理多摄像头图像，我们实现了动态视图融合机制：

- **视图权重分配**：根据不同视图的信息量动态调整注意力权重
- **上下文一致性**：确保跨视图的物体识别结果保持一致性
- **信息冗余过滤**：去除多视图间的重复信息，提高推理效率

这些修复和改进工作耗费了团队大量精力，但显著提升了DriveLM系统的稳定性和性能。修复后的代码可以正确处理多视图输入，为我们的优化实验提供了可靠的基础。

## 使用说明

### 1. Prompt Engineering优化

运行以下命令测试Prompt Engineering优化效果：

```bash
cd /home/zsn/course/AI-acceleration
python report_files/prompt_engineering_code.py
```

这个脚本会：
- 加载PaliGemma视觉语言模型
- 比较原始提示和优化提示的效果
- 输出Token数量、推理时间和结果对比

### 2. COT步骤减少优化

运行以下命令测试COT步骤减少优化：

```bash
cd /home/zsn/course/AI-acceleration
python report_files/cot_optimization_code.py
```

这个脚本会：
- 对比原始5步COT流程和优化后的3步COT流程
- 测量并输出推理时间、Token消耗和结果质量
- 计算优化带来的性能提升

### 3. 组合优化实现

运行以下命令测试两种优化方法的组合效果：

```bash
cd /home/zsn/course/AI-acceleration
python report_files/combined_optimization.py
```

此脚本会：
- 运行基线多步COT流程(5步)
- 运行综合优化流程(2步)，结合了优化的Prompt和减少的COT步骤
- 计算并展示各项性能指标的改进
- 生成可视化对比图表，保存在`report_files/comparison_results.png`

### 4. 生成报告图表

如果需要重新生成报告中的所有图表：

```bash
cd /home/zsn/course/AI-acceleration
python report_files/figures/create_result_figures.py
```

此脚本会根据报告中的数据生成四张对比图表：
- Prompt优化效果图
- COT优化效果图
- 综合优化效果图
- 各优化方向提升百分比对比图

所有图表会保存到`report_files/figures/`目录。

## 注意事项

1. **模型加载**：代码默认使用`google/paligemma-3b-pt-224`模型，但可能需要访问权限或本地下载模型权重。
   
2. **图像路径**：在运行脚本前，请确保图像路径正确。默认使用的是`./assignment/DriveVLMs_v2/demos/samples/`目录下的图像。

3. **字体配置**：可视化代码会自动检测系统中可用的字体，如果没有找到支持中文的字体，将使用英文标签。

4. **资源需求**：运行这些脚本需要一定的GPU资源，特别是加载大型视觉语言模型时。

## 多视图图像处理示例

对于多视图图像处理，我们的修复代码示例：

```python
def process_multi_view_images(image_paths, prompt, processor, model):
    """
    修复后的多视图图像处理函数
    """
    # 加载6个摄像头视图图像
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # 使用优化后的tokenize函数正确处理多图像输入
    inputs = tokenize([prompt], images, processor)
    
    # 模型推理
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.0
    )
    
    # 解码输出
    decoded_output = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    return decoded_output
```

## 模型输出示例

### Prompt优化示例输出
```
=== 原始提示 ===
Token 数量: 312
推理时间: 1.850秒
结果: 在当前场景中有以下重要物体...

=== 优化提示 ===
Token 数量: 187
推理时间: 1.210秒
结果: <c1,CAM_FRONT,450,320>...
```

### COT优化示例输出
```
===== 运行原始多步COT (5步) =====
步骤 1 完成，耗时 1.345秒，生成 285 个token
...
步骤 5 完成，耗时 1.420秒，生成 310 个token

===== 运行优化后COT (3步) =====
步骤 1 完成，耗时 1.542秒，生成 345 个token
...
步骤 3 完成，耗时 1.325秒，生成 298 个token
```

## 扩展与修改

您可以通过以下方式扩展或修改代码：

1. **测试新的提示模板**：修改`prompt_engineering_code.py`中的`OPTIMIZED_PROMPT`变量。

2. **尝试不同的COT步骤合并**：修改`cot_optimization_code.py`中的`OPTIMIZED_COT_PROMPTS`变量。

3. **测试其他VLM模型**：修改`load_model_and_processor`函数中的`model_path`参数。

4. **使用自定义数据**：更改脚本中的`image_paths`列表，指向您自己的图像文件。

## 故障排除

1. **模型加载失败**：确保有正确的模型访问权限，或考虑使用较小的本地模型。

2. **图像加载错误**：检查图像路径是否正确，文件是否存在。

3. **CUDA内存不足**：降低批处理大小或使用更小的模型。

4. **字体渲染问题**：如果图表中的中文无法正确显示，代码会自动切换到英文标签。 