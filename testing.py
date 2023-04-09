import torch
import transformers
from peft import (LoraConfig, get_peft_model, set_peft_model_state_dict)
from transformers import LlamaForCausalLM, LlamaTokenizer

base_model: str = "huggyllama/llama-7b"
checkpoint_name = "./lora_weights/checkpoint-20"
resume_training = True

tokenizer = tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto")

peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")

if resume_training:
    checkpoint_path = f"{checkpoint_name}/pytorch_model.bin"
    peft_config.inference_mode = False
    adapter_weights = torch.load(checkpoint_path)

model = get_peft_model(model, peft_config)
set_peft_model_state_dict(model, adapter_weights)
model.print_trainable_parameters()

trainer = transformers.Trainer(
        model=model,
        train_dataset="I really like tacos")
trainer.train(resume_from_checkpoint=checkpoint_name)