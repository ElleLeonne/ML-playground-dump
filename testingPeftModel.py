import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict)
from transformers import LlamaForCausalLM, LlamaTokenizer

base_model: str = "huggyllama/llama-7b"
checkpoint_name = "./lora_weights/checkpoint-20"
resume_training = True

tokenizer = tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto")

if resume_training:
    checkpoint_path = f"{checkpoint_name}/pytorch_model.bin"
    adapter_weights = torch.load(checkpoint_path)
    model = PeftModel.from_pretrained(model, "./lora_weights", is_trainable=True)

#model = get_peft_model(model, peft_config)
set_peft_model_state_dict(model, adapter_weights)
model.print_trainable_parameters()

trainer = transformers.Trainer(
        model=model,
        train_dataset="I really like tacos")
trainer.train(resume_from_checkpoint=checkpoint_name)