import os

os.environ["HF_HOME"] = "/data2/liguosen/llm/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from datasets import Dataset
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import re


class CleanMemorySFTTrainer(SFTTrainer):
    def evaluate(self, *args, **kwargs):
        # 1. 正常执行 eval（会占用较大显存）
        result = super().evaluate(*args, **kwargs)

        # 2. 强制同步（避免残留 CUDA kernel）
        torch.cuda.synchronize()

        # 3. 清理 PyTorch CUDA allocator 中的碎片块
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("🚀 Eval 后显存已清理，继续训练不会爆显存。")

        return result


def clean_text(text: str) -> str:
    """
    统一清洗文本：
    1. 去掉首尾空格
    2. 去掉所有换行 / 回车
    3. 将连续空白字符压缩为 1 个空格
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = text.replace("\r", "").replace("\n", "").replace(" ", "")
    text = re.sub(r"\s+", "", text)

    return text


# 这里加载第一阶段微调的模型，需要修改成自己的路径
model_id = "/data2/liguosen/llm/med/deepseek-1.5b-data1-3epochs/checkpoint-3"

device = "cuda" if torch.cuda.is_available() else "cpu"

#  数据加载（第二阶段数据集）
csv_path = "data2-100.csv"

df = pd.read_csv(csv_path)

# 新增固定指令列
INSTRUCTION = """你是一名医学专家，需要纠正胃肠道内窥镜检查中“诊断记录”的内容。
要求：
1. 判断输入文本中是否存在错误（不超过三处）。
2. 错误类型仅包括：
   - 插入错误：意外加入错误的单词或表达方式，如用词不当、错误的词语替换、插入不必要的内容或词语混淆（例如，用 “异常” 代替 “正常”）；
   - 遗漏错误：遗漏相关的单词或表达方式，包括删除和缺词（例如，用 “萎缩性胃炎” 代替 “非萎缩性胃炎”）；
   - 部位混淆：在指明正确的解剖位置时出现错误（例如，用 “食管” 代替 “胃窦”；用 “升结肠” 代替 “降结肠”）；
   - 拼写错误：错别字或轻微拼写偏差，包括单词或短语的截断；
   - 其他错误：不属于上述类别的错误，包括输入错误的日期、图像和/或序列编号错误、计量单位错误、模板错误和标点符号错误。
3. 输入的错误文本中存在一些部位混淆错误，而一般情况下胃肠道内窥镜“诊断记录”的是按照部位的顺序书写：
   “食管......。贲门......。胃底......。胃体......。胃角......。胃窦......。幽门......。十二指肠球部......。十二指肠降段......。“
4. 按照以下步骤去思考纠正这些错误：
   （1）错误定位：找到错误的词组；
   （2）错误类型判断：对存在错误的词组进行错误分类；
   （3）错误纠正：在前述分析基础上输出纠正文本。
5. 最后请严格遵守以下格式输出：
### 输出纠正后的文本
正确文本：xxxxxx

"""
# -------------------- 构造训练数据 --------------------
records = []
for _, row in df.iterrows():
    input_text = clean_text(row["错误文本"])
    correct_text = clean_text(row["正确文本"])

    records.append({
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": correct_text,  # 纠正后的文本
    })

df_final = pd.DataFrame(records)
dataset = Dataset.from_pandas(df_final)

# ======================
# 划分训练集和验证集（9:1）
# ======================
dataset = dataset.shuffle(seed=42)  # 随机打乱，保证每次划分不同
split_dataset = dataset.train_test_split(test_size=0.10)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(eval_dataset)}")


# -------------------- 模型与分词器 --------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# -------------------- LoRA 配置 --------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)


# -------------------- 拼接输入输出格式 --------------------
def formatting_prompts_func(example):
    corrected_text = example["output"]

    final_answer = f"""
### 输出纠正后的文本
正确文本：{corrected_text}
""".strip()

    text = (
        f"<|im_start|>user\n"
        f"{example['instruction']}\n"
        f"输入文本：{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{final_answer}<|im_end|>"
    )
    return {"text": text}


train_dataset = train_dataset.map(formatting_prompts_func, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(formatting_prompts_func, remove_columns=eval_dataset.column_names)

# SFT 配置
num_train_epochs = 3
output_dir = f"deepseek-1.5b-data2-{num_train_epochs}epochs"

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    max_seq_length=1100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=10,
    logging_first_step=True,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    packing=False,
    dataset_text_field="text",
    remove_unused_columns=True,
    report_to=["tensorboard"],
    # ✅ 新增验证配置
    eval_strategy="steps",
    eval_steps=200,  # 验证频率（可根据训练规模调整）
    label_names=["labels"]
)

# Trainer
trainer = CleanMemorySFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=sft_config,
)

# 启动训练
trainer.train()

# 保存最终模型
trainer.save_model()
