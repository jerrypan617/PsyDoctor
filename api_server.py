import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn

app = FastAPI(title="PsyDoctor API", description="心理咨询师对话API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_length: Optional[int] = 1024

class ChatResponse(BaseModel):
    response: str
    role: str = "assistant"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()
print(f"使用设备: {device}")

def load_model():
    """加载模型和tokenizer"""
    global model, tokenizer
    
    print("加载基模型...")
    if device == "cuda":
        device_map = "auto"
    elif device == "mps":
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        dtype=torch.bfloat16,
        device_map=device_map,
        cache_dir="base_model"
    )
    
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./output")
    
    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, "./output/lora_adapter")
    
    if device == "mps":
        model = model.to("mps")
    elif device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    
    print("模型加载成功。")

def format_conversation(messages):
    """格式化对话"""
    formatted_text = ""
    for message in messages:
        role = message.role if isinstance(message, ChatMessage) else message["role"]
        content = message.content if isinstance(message, ChatMessage) else message["content"]
        
        if role == "system":
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return formatted_text

def generate_response(messages, max_length=1024, temperature=0.7):
    """生成回复"""
    input_text = format_conversation(messages)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    elif device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
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

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    print("正在启动服务器...")
    load_model()
    print("服务器已就绪！")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "PsyDoctor API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        messages = request.messages
        has_system = any(msg.role == "system" for msg in messages)
        if not has_system:
            system_message = ChatMessage(
                role="system",
                content="你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。"
            )
            messages = [system_message] + messages
        
        response_text = generate_response(
            messages,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回复时出错: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )