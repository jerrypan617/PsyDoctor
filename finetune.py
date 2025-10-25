import os
import json
import random
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = "/root/autodl-tmp/bs/models"
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    use_auth_token: bool = False

@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = "/root/autodl-tmp/bs/datas/PsyDTCorpus_train_mulit_turn_packing.json"
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    train_val_split: float = 0.9

@dataclass
class TrainingArguments:
    """训练相关参数"""
    output_dir: str = "/root/autodl-tmp/bs/output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "tensorboard"
    logging_dir: str = "/root/tf-logs"
    run_name: str = "qwen2-lora-finetune"
    fp16: bool = True
    gradient_checkpointing: bool = True  # 启用梯度检查点
    optim: str = "paged_adamw_8bit"  # 使用8bit优化器
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    max_grad_norm: float = 1.0  # 梯度裁剪

@dataclass
class LoRAArguments:
    """LoRA相关参数"""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"

class MultiTurnDataset(Dataset):
    """多轮对话数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        
        # 构建对话文本
        text = self._format_conversation(messages)
        
        # 编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None  # 不返回tensor，返回list
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # 标签与输入相同
        labels = input_ids.copy()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """格式化多轮对话"""
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

class DataCollatorForCausalLM:
    """自定义数据整理器，用于因果语言模型"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features):
        # 获取batch中的最大长度
        max_length = max(len(f["input_ids"]) for f in features)
        
        # 如果指定了pad_to_multiple_of，调整max_length
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]
            
            padding_length = max_length - len(input_ids)
            
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
            labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        batch = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }
        
        return batch

def load_data(data_path: str, train_val_split: float = 0.9):
    """加载和划分数据"""
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 划分训练集和验证集
    split_idx = int(len(data) * train_val_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def setup_lora_model(model, lora_args: LoRAArguments):
    """设置LoRA模型"""
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def compute_metrics(eval_preds):
    """计算评估指标"""
    predictions, labels = eval_preds
    # 忽略-100的标签
    predictions = np.argmax(predictions, axis=-1)
    
    # 只计算非-100标签的准确率
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
    }

class LoRATrainer(Trainer):
    """自定义Trainer，只保存LoRA适配器"""
    
    def save_model(self, output_dir=None, _internal_call=False):
        """重写保存方法，只保存LoRA适配器"""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # 只保存LoRA适配器
        if hasattr(self.model, 'save_pretrained'):
            lora_path = os.path.join(output_dir, "lora_adapter")
            os.makedirs(lora_path, exist_ok=True)
            self.model.save_pretrained(lora_path)
            
            # 保存LoRA配置
            if hasattr(self.model, 'peft_config'):
                lora_config = self.model.peft_config
                with open(os.path.join(lora_path, "lora_config.json"), "w") as f:
                    json.dump(lora_config, f, indent=2)
            
            print(f"LoRA adapter saved to {lora_path}")
        else:
            print("Model does not have save_pretrained method")

def main():
    set_seed(42)
    
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArguments()
    lora_args = LoRAArguments()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)
    logger.info(f"TensorBoard logs will be saved to: {training_args.logging_dir}")
    logger.info(f"To view logs, run: tensorboard --logdir={training_args.logging_dir}")
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    logger.info("Setting up LoRA...")
    model = setup_lora_model(model, lora_args)
    
    logger.info("Loading data...")
    train_data, val_data = load_data(data_args.data_path, data_args.train_val_split)
    
    train_dataset = MultiTurnDataset(train_data, tokenizer, data_args.max_seq_length)
    val_dataset = MultiTurnDataset(val_data, tokenizer, data_args.max_seq_length)
    
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )
    
    # 训练参数
    from transformers import TrainingArguments as HFTrainingArguments
    
    hf_training_args = HFTrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        eval_strategy=training_args.evaluation_strategy,
        save_strategy="no",
        save_total_limit=0,
        load_best_model_at_end=False,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        report_to=training_args.report_to,
        logging_dir=training_args.logging_dir,
        run_name=training_args.run_name,
        fp16=training_args.fp16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        optim=training_args.optim,
        max_grad_norm=training_args.max_grad_norm,
        dataloader_pin_memory=training_args.dataloader_pin_memory,
        remove_unused_columns=training_args.remove_unused_columns,
        eval_accumulation_steps=4,  # 评估时也使用梯度累积
    )
    
    # 创建训练器
    trainer = LoRATrainer(
        model=model,
        args=hf_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving final model...")
    model.save_pretrained(os.path.join(hf_training_args.output_dir, "lora_adapter"))
    tokenizer.save_pretrained(hf_training_args.output_dir)
    lora_config = model.peft_config
    with open(os.path.join(hf_training_args.output_dir, "lora_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"View training logs with: tensorboard --logdir={training_args.logging_dir}")

if __name__ == "__main__":
    main()