import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model():
    print("加载基模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "models",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/bs/output")
    
    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, "/root/autodl-tmp/bs/output/lora_adapter")
    
    print("模型加载成功。")
    return model, tokenizer

def format_conversation(messages):
    """格式化对话"""
    formatted_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return formatted_text

def generate_response(model, tokenizer, messages, max_length=1024, temperature=0.7):
    """生成回复"""
    input_text = format_conversation(messages)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)
    input_length = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = response.strip()
    if response.startswith("assistant"):
        first_newline = response.find('\n')
        if first_newline != -1:
            response = response[first_newline + 1:].strip()
        else:
            response = response[8:].strip()
    if response.endswith("<|im_end|>"):
        response = response[:-10].strip()
    
    return response

def main():
    model, tokenizer = load_model()
    messages = [
        {
            "role": "system",
            "content": "你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。"
        }
    ]
    while True:
        try:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            print("助手: ", end="", flush=True)
            response = generate_response(model, tokenizer, messages)
            print(response)
            messages.append({"role": "assistant", "content": response})
            if len(messages) > 10:
                messages = messages[:1] + messages[-9:]
        except KeyboardInterrupt:
            print("\n\n程序被Ctrl+C中断，再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue
if __name__ == "__main__":
    main()