import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# 定义原始和优化后的提示模板
ORIGINAL_PROMPT = """answer en You are an driving labeler. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>)
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

OPTIMIZED_PROMPT = """answer en You are an autonomous driving system analyzer with six camera views. 
Front camera <image> shows the forward view.
Front-left camera <image> shows the left diagonal view.
Front-right camera <image> shows the right diagonal view.
Back camera <image> shows the rear view.
Back-left camera <image> shows the left rear diagonal view.
Back-right camera <image> shows the right rear diagonal view.

### Instruction:
{instruction}

### Response:"""

# 针对不同任务的优化指令
PERCEPTION_INSTRUCTION_ORIGINAL = "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."
PERCEPTION_INSTRUCTION_OPTIMIZED = """Identify the key objects in the current scene that are relevant for driving decisions. For each object, provide:
1. Object type (vehicle, pedestrian, cyclist, etc.)
2. Camera view where it's visible
3. Approximate location (x,y coordinates)
Format each object as: <object_id,camera_name,x_coord,y_coord>"""

PREDICTION_INSTRUCTION_ORIGINAL = "What is the moving status of object <c1,CAM_BACK,384.2,477.5>? Please describe in detail."
PREDICTION_INSTRUCTION_OPTIMIZED = "What is the moving status of object <c1,CAM_BACK,384.2,477.5>? Choose from: [moving forward, turning left, turning right, stopped, reversing]"

PLANNING_INSTRUCTION_ORIGINAL = "What should the ego vehicle do next? Please provide a detailed plan."
PLANNING_INSTRUCTION_OPTIMIZED = "What should the ego vehicle do next? Provide your answer in this format: [action: X, speed: Y, reason: Z]"

def format_prompt(template, instruction):
    """格式化提示模板"""
    return template.format(instruction=instruction)

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

def infer(model, inputs, max_new_tokens=512):
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
    return output

def run_comparison(image_paths, model, processor):
    """比较原始提示和优化提示的效果"""
    results = {}
    
    # 加载图像
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # 原始感知提示测试
    original_prompt = format_prompt(ORIGINAL_PROMPT, PERCEPTION_INSTRUCTION_ORIGINAL)
    original_inputs = tokenize([original_prompt], images, processor)
    
    # 计时开始
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    original_output = infer(model, original_inputs)
    end_time.record()
    torch.cuda.synchronize()
    original_time = start_time.elapsed_time(end_time) / 1000  # 转换为秒
    
    original_results = processor.batch_decode(original_output, skip_special_tokens=True)
    original_token_count = len(original_output[0])
    
    # 优化感知提示测试
    optimized_prompt = format_prompt(OPTIMIZED_PROMPT, PERCEPTION_INSTRUCTION_OPTIMIZED)
    optimized_inputs = tokenize([optimized_prompt], images, processor)
    
    start_time.record()
    optimized_output = infer(model, optimized_inputs)
    end_time.record()
    torch.cuda.synchronize()
    optimized_time = start_time.elapsed_time(end_time) / 1000  # 转换为秒
    
    optimized_results = processor.batch_decode(optimized_output, skip_special_tokens=True)
    optimized_token_count = len(optimized_output[0])
    
    # 记录结果
    results = {
        "original": {
            "prompt": original_prompt,
            "result": original_results[0],
            "token_count": original_token_count,
            "inference_time": original_time
        },
        "optimized": {
            "prompt": optimized_prompt,
            "result": optimized_results[0],
            "token_count": optimized_token_count,
            "inference_time": optimized_time
        }
    }
    
    return results

def print_comparison_results(results):
    """打印比较结果"""
    print("=== 原始提示 ===")
    print(f"Token 数量: {results['original']['token_count']}")
    print(f"推理时间: {results['original']['inference_time']:.3f}秒")
    print(f"结果: {results['original']['result'][:100]}...")
    
    print("\n=== 优化提示 ===")
    print(f"Token 数量: {results['optimized']['token_count']}")
    print(f"推理时间: {results['optimized']['inference_time']:.3f}秒")
    print(f"结果: {results['optimized']['result'][:100]}...")
    
    # 计算改进百分比
    token_improvement = (1 - results['optimized']['token_count'] / results['original']['token_count']) * 100
    time_improvement = (1 - results['optimized']['inference_time'] / results['original']['inference_time']) * 100
    
    print("\n=== 改进百分比 ===")
    print(f"Token 减少: {token_improvement:.1f}%")
    print(f"推理时间减少: {time_improvement:.1f}%")

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
    results = run_comparison(image_paths, model, processor)
    print_comparison_results(results) 