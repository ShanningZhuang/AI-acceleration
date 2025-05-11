import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM
from peft import PeftModel
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
import argparse
from drivevlms.build import build_collate_fn
import json

@torch.no_grad() 
def main(args):

    # Load model and processor
    processor = Phi4MMProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct",  revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")
    model = Phi4MMForCausalLM.from_pretrained('phi-4-multimodal-finetuned',
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa')
    model.to(args.device)
    generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

    # prepare dataset
    collate_fn = build_collate_fn(args.collate_fn)
    val_collate_fn = partial(collate_fn, processor=processor, dtype=torch.float16)
    dataset = load_from_disk(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=val_collate_fn,
        num_workers=16,
        shuffle=False,
    )

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

    def flatten(x):
        return x[0] if isinstance(x, list) else x
    
    data_dict = []
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader):
            cnt += 1
            inputs, question, ids = batch
            breakpoint()
            results = infer(inputs.to(args.device))
            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference')
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    parser.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_results_21-49.json")
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())