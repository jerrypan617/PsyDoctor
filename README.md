# PsyDoctor

## Weights

The LoRA Adapter's weight is available on Huggingface:

```
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "JERRYPAN617/qwen2.5-lora-psydoctor")
```

Support inference on Apple MPS and Nvidia CUDA.

## Tasks:

1. LoRA Fine-tuned Qwen2.5-1.5B-Instruct (Adapter) on PsyDTCorpus dataset. (Done)