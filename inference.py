import os
import re
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\r", "").replace("\n", "").replace(" ", "")
    text = re.sub(r"\s+", "", text)
    return text


# 基础模型路径
base_model = "/data2/liguosen/llm/base_model/deepseek-1.5b"
# 第二阶段微调后的 checkpoint 路径
lora_model = "/data2/liguosen/llm/med/deepseek-1.5b-data2-3epochs/checkpoint-9"
# 推理使用的 instruction（和训练时保持一致）
instruction = """你是一名医学专家，需要纠正胃肠道内窥镜检查中“诊断记录”的内容。
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

# -------------------- 加载模型与分词器 --------------------
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_model)
model.eval()


# -------------------- 推理函数 --------------------
def infer_and_parse(error_text):
    """根据输入的错误文本，生成模型输出并解析为JSON"""
    prompt = (
        f"<|im_start|>user\n{instruction}\n输入文本：{error_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.9,
        do_sample=True
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # print(decoded)
    # 提取 assistant 的回答部分
    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in decoded:
        decoded = decoded.split("<|im_end|>", 1)[0]
    # 清除所有 ASCII 和 Unicode 控制字符
    decoded = re.sub(
        r"[\u0000-\u001F\u007F-\u009F\u200E\u200F\u202A-\u202E]",
        "",
        decoded
    ).strip()

    # -------------------- 解析 “正确文本” --------------------
    correct_text = ""
    m = re.search(r"正确文本[:：]\s*(.*)", decoded)
    if m: correct_text = m.group(1).strip()
    print("===============================================")
    print(correct_text)

    return decoded, correct_text


# -------------------- 读取测试数据 --------------------
csv_path = "test.csv"

df = pd.read_csv(csv_path)

# -------------------- 批量推理 --------------------
# raw_outputs = []
correct_texts = []

for idx, row in df.iterrows():
    error_text = clean_text(str(row["错误文本"]))
    print(f"[{idx + 1}/{len(df)}] 推理中...")

    raw_out, corr = infer_and_parse(error_text)

    correct_texts.append(corr)


# -------------------- 保存结果 --------------------
df["正确文本-模型"] = correct_texts

output_path = "test_output.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 推理完成，结果已保存到：{output_path}")
