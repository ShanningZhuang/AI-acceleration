import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
from peft import PeftModel

# 定义原始的多步COT提示模板
ORIGINAL_COT_PROMPTS = [
    # 第一步：场景感知
    """answer en You are an autonomous driving assistant. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>).

### Instruction:
What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.

### Response:""",

    # 第二步：车辆运动状态分析
    """answer en You are an autonomous driving assistant. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>).

### Instruction:
What is the moving status of object <c1,CAM_BACK,384.2,477.5>? Is it moving, stopped, turning?

### Response:""",

    # 第三步：行人意图分析
    """answer en You are an autonomous driving assistant. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>).

### Instruction:
What is the intent of the pedestrian <p1,CAM_FRONT,220.5,310.8>? Is the pedestrian crossing the road or waiting?

### Response:""",

    # 第四步：路径规划
    """answer en You are an autonomous driving assistant. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>).

### Instruction:
Based on the scene analysis where we have identified a vehicle <c1,CAM_BACK,384.2,477.5> that is moving forward and a pedestrian <p1,CAM_FRONT,220.5,310.8> who is about to cross the road, what driving path should the ego vehicle take?

### Response:""",

    # 第五步：最终驾驶决策
    """answer en You are an autonomous driving assistant. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>).

### Instruction:
Based on all previous analysis, what's the final driving decision? Should the ego vehicle accelerate, maintain speed, slow down, or stop? And in which direction should it steer?

### Response:"""
]

# 定义优化后的减少COT步骤的提示模板
OPTIMIZED_COT_PROMPTS = [
    # 第一步：综合场景感知和物体状态分析
    """answer en You are an autonomous driving system with six camera views.
Front camera <image>, front-left camera <image>, front-right camera <image>, back camera <image>, back-left camera <image>, and back-right camera <image> show the surrounding environment.

### Instruction:
Analyze the driving scene comprehensively:
1. Identify all important objects (vehicles, pedestrians, cyclists, etc.)
2. For each object, specify its location as <object_id,camera_name,x_coord,y_coord>
3. Describe the motion status of each object (moving, stopped, turning, etc.)

### Response:""",

    # 第二步：风险预测和规划
    """answer en You are an autonomous driving system with six camera views.
Front camera <image>, front-left camera <image>, front-right camera <image>, back camera <image>, back-left camera <image>, and back-right camera <image> show the surrounding environment.

### Instruction:
Based on the scene analysis where we have identified various objects and their motion status:
1. Predict the potential risks in the next 3 seconds
2. Propose a driving path that avoids all identified risks
Be specific about which objects pose the greatest concern and why.

### Response:""",

    # 第三步：最终决策
    """answer en You are an autonomous driving system with six camera views.
Front camera <image>, front-left camera <image>, front-right camera <image>, back camera <image>, back-left camera <image>, and back-right camera <image> show the surrounding environment.

### Instruction:
Based on the scene analysis and risk assessment, provide the final driving decision in this format:
- Action: [accelerate/maintain/slow down/stop]
- Steering: [straight/slight left/moderate left/sharp left/slight right/moderate right/sharp right]
- Speed: [target speed in km/h]
- Reason: [brief justification for this decision]

### Response:"""
]

def load_model_and_processor(model_path="google/paligemma-3b-pt-224", lora_path=None):
    """加载模型和处理器"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path)
    
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    
    model.to('cuda')
    return model, processor

def tokenize(texts, images, processor, device='cuda'):
    """将文本和图像处理为模型输入"""
    return processor(
        text=texts, images=images, return_tensors="pt", padding="longest"
    ).to(device)

def infer(model, inputs, processor, max_new_tokens=512):
    """模型推理函数"""
    input_len = inputs["input_ids"].shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.5,
        repetition_penalty=1.02
    )
    output = output[:, input_len:]
    results = processor.batch_decode(output, skip_special_tokens=True)
    return results[0], len(output[0])

def run_original_cot(image_paths, model, processor):
    """运行原始多步COT流程"""
    results = []
    total_time = 0
    total_tokens = 0
    
    # 加载图像
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    for i, prompt in enumerate(ORIGINAL_COT_PROMPTS):
        # 构建输入
        inputs = tokenize([prompt], images, processor)
        
        # 计时
        start_time = time.time()
        result, tokens = infer(model, inputs, processor)
        end_time = time.time()
        
        # 记录结果
        inference_time = end_time - start_time
        total_time += inference_time
        total_tokens += tokens
        
        results.append({
            "step": i+1,
            "prompt": prompt,
            "result": result,
            "tokens": tokens,
            "time": inference_time
        })
        
        print(f"步骤 {i+1} 完成，耗时 {inference_time:.3f}秒，生成 {tokens} 个token")
    
    return results, total_time, total_tokens

def run_optimized_cot(image_paths, model, processor):
    """运行优化后的减少COT步骤流程"""
    results = []
    total_time = 0
    total_tokens = 0
    
    # 加载图像
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    for i, prompt in enumerate(OPTIMIZED_COT_PROMPTS):
        # 构建输入
        inputs = tokenize([prompt], images, processor)
        
        # 计时
        start_time = time.time()
        result, tokens = infer(model, inputs, processor)
        end_time = time.time()
        
        # 记录结果
        inference_time = end_time - start_time
        total_time += inference_time
        total_tokens += tokens
        
        results.append({
            "step": i+1,
            "prompt": prompt,
            "result": result,
            "tokens": tokens,
            "time": inference_time
        })
        
        print(f"步骤 {i+1} 完成，耗时 {inference_time:.3f}秒，生成 {tokens} 个token")
    
    return results, total_time, total_tokens

def compare_cot_approaches(image_paths, model, processor):
    """比较原始COT和优化COT的性能"""
    print("===== 运行原始多步COT (5步) =====")
    original_results, original_time, original_tokens = run_original_cot(image_paths, model, processor)
    
    print("\n===== 运行优化后COT (3步) =====")
    optimized_results, optimized_time, optimized_tokens = run_optimized_cot(image_paths, model, processor)
    
    # 计算性能提升
    time_improvement = (1 - optimized_time / original_time) * 100
    token_improvement = (1 - optimized_tokens / original_tokens) * 100
    
    print("\n===== 性能比较 =====")
    print(f"原始COT (5步):")
    print(f"  - 总推理时间: {original_time:.3f}秒")
    print(f"  - 总生成token数: {original_tokens}")
    
    print(f"\n优化COT (3步):")
    print(f"  - 总推理时间: {optimized_time:.3f}秒")
    print(f"  - 总生成token数: {optimized_tokens}")
    
    print(f"\n性能提升:")
    print(f"  - 推理时间减少: {time_improvement:.1f}%")
    print(f"  - Token消耗减少: {token_improvement:.1f}%")
    
    return {
        "original": {
            "results": original_results,
            "total_time": original_time,
            "total_tokens": original_tokens
        },
        "optimized": {
            "results": optimized_results,
            "total_time": optimized_time,
            "total_tokens": optimized_tokens
        },
        "improvements": {
            "time": time_improvement,
            "tokens": token_improvement
        }
    }

if __name__ == "__main__":
    # 图像路径
    image_paths = [
        "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_FRONT__1537295990612404.jpg",
        "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_FRONT_LEFT__1537295990604799.jpg",
        "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_FRONT_RIGHT__1537295990620482.jpg",
        "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_BACK__1537295990637558.jpg",
        "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_BACK_LEFT__1537295990647405.jpg",
        "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_BACK_RIGHT__1537295990628113.jpg",
    ]
    
    # 加载模型
    model, processor = load_model_and_processor(
        model_path="google/paligemma-3b-pt-224",
        lora_path="drivelm-project/paligemma-finetuned-lora"
    )
    
    # 运行对比实验
    comparison_results = compare_cot_approaches(image_paths, model, processor) 