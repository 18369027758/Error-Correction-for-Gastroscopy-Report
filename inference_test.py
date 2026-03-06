import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
model_path = "/data2/liguosen/llm/base_model/deepseek-1.5b"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 输入问题
prompt = "你好，你是谁？"

# 编码
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 解码
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("模型回答：")
print(response)