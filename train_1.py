import os

os.environ["HF_HOME"] = "/data2/liguosen/llm/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from datasets import Dataset
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 模型
model_id = "/data2/liguosen/llm/base_model/deepseek-1.5b"

# 加载本地 CSV 数据
csv_path = "data1-200.csv"  # 替换为你的 CSV 路径
df = pd.read_csv(csv_path)
# 新增固定指令列
df["instruction"] = "下面句子中存在一些拼写或语法错误，请纠正句子中的错误，并返回正确句子"
df = df.rename(columns={"错误句子": "input", "正确句子": "output"})
df = df[["instruction", "input", "output"]]

# 转换成 HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# ======================
# 划分训练集和验证集（9:1）
# ======================
dataset = dataset.shuffle(seed=42)  # 随机打乱，保证每次划分不同
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(eval_dataset)}")

# Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)


# 数据格式化函数
def formatting_prompts_func(example: dict) -> str:
    # 拼接 instruction + input + output
    return f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"


# SFT 配置
num_train_epochs = 3
output_dir = f"{model_id.split('/')[-1]}-data1-{num_train_epochs}epochs"

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    max_seq_length=256,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    logging_first_step=True,
    learning_rate=1e-3,
    weight_decay=0.01,
    fp16=False,
    bf16=True,  # 确保 GPU 支持
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    packing=True,
    report_to=["tensorboard"],
    # ✅ 新增验证配置
    eval_strategy="steps",
    eval_steps=100,  # 验证频率（可根据训练规模调整）
    label_names=["labels"],
    save_total_limit=10,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    args=sft_config,
)

# 启动训练
trainer.train()

# 保存最终模型
trainer.save_model()
