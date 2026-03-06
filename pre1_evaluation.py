import pandas as pd
import json
import difflib


def clean_text(s: str):
    """去掉空格和换行"""
    return s.replace(" ", "").replace("\n", "").replace("\r", "")


def find_differences(a, b):
    """
    使用 difflib 找出两段文本的差异
    返回列表：[{ "corr": X, "erro": Y, "start": i, "end": j }]
    """
    diff = []
    seq = difflib.SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag == "equal":
            continue

        corr = a[i1:i2]  # 正确文本片段
        erro = b[j1:j2]  # 错误文本片段

        # 跳过空的情况
        if corr == "" and erro == "":
            continue

        diff.append({
            "corr": corr,
            "erro": erro,
            "type": "diff自动识别",
            "start": i1,
            "end": i2
        })
    return diff


def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    updated_records = []

    for idx, row in df.iterrows():
        correct = str(row["正确文本"])
        error = str(row["错误文本"])

        # 清理文本
        clean_correct = clean_text(correct)
        clean_error = clean_text(error)

        # 直接比对差异，不再使用“具体错误-模型”
        diffs = find_differences(clean_correct, clean_error)

        # 保存到新列
        row["具体错误-原始"] = json.dumps(diffs, ensure_ascii=False)

        updated_records.append(row)

        print(f"处理第 {idx + 1} 行，差异数量：{len(diffs)}")

    pd.DataFrame(updated_records).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成！已保存到：{output_file}")


process_csv(
    "测试数据-1k-方案C_模型3.csv",
    "测试数据-1k-方案C_模型3_原始.csv"
)
