import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score


def normalize(text):
    """去除空格和标点"""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\s，。、“”‘’；：,.!?！？]", "", text)


def is_correction_valid(gt_corr, gt_erro, pred_corr, pred_erro):
    """
    判断模型纠错是否正确：
    ✅ 完全匹配；
    ✅ 字数不同 → 根据长短方向判断差异部分是否合理；
    """
    gt_corr, gt_erro = normalize(gt_corr), normalize(gt_erro)
    pred_corr, pred_erro = normalize(pred_corr), normalize(pred_erro)

    # 1. 完全匹配
    if gt_corr == pred_corr and gt_erro == pred_erro:
        return True

    # 2. 字数不同
    if len(pred_corr) != len(gt_corr):
        # # 子串关系判断
        # if gt_corr in pred_corr or pred_corr in gt_corr:
        #     # pred_corr 比 gt_corr 长
        #     if len(pred_corr) > len(gt_corr):
        #         diff_part = pred_corr.replace(gt_corr, "")
        #         if diff_part and diff_part not in pred_erro:
        #             return False
        #         else:
        #             return True
        #     else:
        #         # pred_corr 比 gt_corr 短
        #         diff_part = gt_corr.replace(pred_corr, "")
        #         if diff_part and diff_part not in gt_erro:
        #             return False
        #         else:
        #             return True
        # else:
        #     return False
        return False

    # 3. 字数相等但不完全匹配
    return False


def match_errors_with_tolerance(gt_list, pred_list, tolerance=0):
    """
    将预测错误与GT错误进行匹配，基于start位置，允许±tolerance偏移
    返回匹配对列表 [(gt_error, pred_error)]
    """
    matched_pairs = []
    used_pred_indices = set()

    for gt in gt_list:
        gt_start = gt.get("start", -999)
        best_match = None
        min_diff = float("inf")

        # 遍历所有预测错误，找出最接近的start
        for j, pred in enumerate(pred_list):
            if j in used_pred_indices:
                continue
            pred_start = pred.get("start", -999)
            diff = abs(pred_start - gt_start)
            if diff <= tolerance and diff < min_diff:
                best_match = (gt, pred)
                min_diff = diff

        if best_match:
            matched_pairs.append(best_match)
            used_pred_indices.add(pred_list.index(best_match[1]))

    return matched_pairs


def evaluate_edit_level(csv_path, target_row=None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["具体错误-原始", "具体错误-模型"])

    type_y_true, type_y_pred = [], []
    corr_y_true, corr_y_pred = [], []

    # ★ 如果指定行号，则只取这一行
    if target_row is not None:
        df = df.iloc[[target_row]]  # 保持 DataFrame 结构，方便后续统一处理逻辑

    for i, row in df.iterrows():
        try:
            gt_list = json.loads(row["具体错误-原始"])
            pred_list = json.loads(row["具体错误-模型"])
            # print(gt_list, pred_list)
        except Exception:
            continue

        # === 🧩 处理空列表情况 ===
        if isinstance(gt_list, list) and isinstance(pred_list, list):
            # 情况1：GT 和 Pred 都为空 → 跳过
            if len(gt_list) == 0 and len(pred_list) == 0:
                continue
            # 情况2：GT 不为空但 Pred 为空 → 全部计为 FN
            elif len(gt_list) > 0 and len(pred_list) == 0:
                corr_y_true.extend([1] * len(gt_list))
                corr_y_pred.extend([0] * len(gt_list))
                continue
            # 情况3：GT 为空但 Pred 不为空 → 全部计为 FP
            elif len(gt_list) == 0 and len(pred_list) > 0:
                corr_y_true.extend([0] * len(pred_list))
                corr_y_pred.extend([1] * len(pred_list))
                continue

        # === 正常匹配 ===
        matched_pairs = match_errors_with_tolerance(gt_list, pred_list, tolerance=2)

        # 🧩 特殊情况：两个都非空但无匹配
        if len(gt_list) > 0 and len(pred_list) > 0 and len(matched_pairs) == 0:
            # 全部算作互相独立的错误
            # GT中未匹配的算 FN
            corr_y_true.extend([1] * len(gt_list))
            corr_y_pred.extend([0] * len(gt_list))
            # Pred中未匹配的算 FP
            corr_y_true.extend([0] * len(pred_list))
            corr_y_pred.extend([1] * len(pred_list))
            continue

        # 1️⃣ 匹配到的错误对
        for gt, pred in matched_pairs:
            gt_type = gt.get("type", "")
            gt_corr, gt_erro = gt.get("corr", ""), gt.get("erro", "")
            pred_type = pred.get("type", "")
            pred_corr, pred_erro = pred.get("corr", ""), pred.get("erro", "")

            # 错误类型准确率
            type_y_true.append(gt_type)
            type_y_pred.append(pred_type)

            # 纠错是否正确
            valid = is_correction_valid(gt_corr, gt_erro, pred_corr, pred_erro)
            corr_y_true.append(1)
            corr_y_pred.append(1 if valid else 0)

        # 2️⃣ GT中未匹配到的错误 → FN
        unmatched_gt_count = len(gt_list) - len(matched_pairs)
        if unmatched_gt_count > 0:
            corr_y_true.extend([1] * unmatched_gt_count)
            corr_y_pred.extend([0] * unmatched_gt_count)

        # 3️⃣ 预测中多出的错误 → FP
        unmatched_pred_count = len(pred_list) - len(matched_pairs)
        if unmatched_pred_count > 0:
            corr_y_true.extend([0] * unmatched_pred_count)
            corr_y_pred.extend([1] * unmatched_pred_count)

        # === 计算指标 ===
    if len(type_y_true) == 0:
        print("没有有效样本，无法计算指标。")
        return
    print(type_y_true, type_y_pred, corr_y_true, corr_y_pred)

    # === 统计 TP / FP / FN ===
    TP = FP = FN = 0
    for y_t, y_p in zip(corr_y_true, corr_y_pred):
        if y_t == 1 and y_p == 1:
            TP += 1
        elif y_t == 0 and y_p == 1:
            FP += 1
        elif y_t == 1 and y_p == 0:
            FN += 1

    type_acc = sum([1 for t, p in zip(type_y_true, type_y_pred) if t == p]) / len(type_y_true)
    corr_prec = precision_score(corr_y_true, corr_y_pred)
    corr_rec = recall_score(corr_y_true, corr_y_pred)
    corr_f1 = f1_score(corr_y_true, corr_y_pred)
    corr_f05 = fbeta_score(corr_y_true, corr_y_pred, beta=0.5)

    print(len(corr_y_true), len(corr_y_pred))
    print("=== 编辑级指标结果 ===")
    print(f"TP（正确纠错）：{TP}")
    print(f"FP（多余纠错）：{FP}")
    print(f"FN（漏纠错）：{FN}")
    # print(f"错误类型准确率：{type_acc:.4f}")
    print(f"纠错精准率 Precision：{corr_prec:.4f}")
    print(f"纠错召回率 Recall：{corr_rec:.4f}")
    print(f"纠错F1分数：{corr_f1:.4f}")
    print(f"纠错F0.5分数：{corr_f05:.4f}")


if __name__ == "__main__":
    csv_path = "测试数据-1k-方案C_模型3.csv"  # 改成你的CSV文件路径
    evaluate_edit_level(csv_path)
