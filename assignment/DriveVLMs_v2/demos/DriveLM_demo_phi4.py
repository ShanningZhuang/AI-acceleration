import torch
from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM
from transformers import GenerationConfig
from PIL import Image
import argparse
import os

_imgs_filename = [
            "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_FRONT__1537295990612404.jpg",
            "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_FRONT_LEFT__1537295990604799.jpg",
            "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_FRONT_RIGHT__1537295990620482.jpg",
            "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_BACK__1537295990637558.jpg",
            "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_BACK_LEFT__1537295990647405.jpg",
            "./demos/samples/n008-2018-09-18-14-35-12-0400__CAM_BACK_RIGHT__1537295990628113.jpg"
        ]
# Load model and processor
processor = Phi4MMProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct",  revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")
model = Phi4MMForCausalLM.from_pretrained(
    'drivelm-project/phi-4-multimodal-finetuned',
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa'
)


model.to('cuda')

# Load generation config
generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

def infer(inputs):

    input_len = inputs["input_ids"].shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config
    )
    output = output[:, input_len:]
    results = processor.batch_decode(output, skip_special_tokens=True)
    return results


def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


def tokenize(texts, images, processor, device='cuda'):
    return processor(
        text=texts, images=images, return_tensors="pt", padding="longest"
    ).to(device)


@torch.no_grad()
def main(args):
    sample_prompts = [
        "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
        "What is the moving status of object <c1,CAM_BACK,384.2,477.5>? Please select the correct answer from the following options: A. Turn right. B. Drive backward. C. Going ahead. D. Turn left."
    ]
    for instruction in sample_prompts:
        prompt = format_prompt(instruction)
        print(f"\nGenerating for prompt: {repr(prompt)}")
        images = [Image.open(cam).convert("RGB") for cam in _imgs_filename]
        images = [image.resize((448, 448), ) for image in images]
        reason_inputs = tokenize([prompt], images, processor, args.device)
        reason_results = infer(reason_inputs)
        print(reason_results)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Phi4 Inference')
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())