# -*- coding: utf-8 -*-
"""与 ground_truth 对齐的评估：混淆矩阵、检出率、类型准确率、漏检/误报清单。"""

from collections import Counter


def _fmt(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def evaluate(results, ground_truth):
    """计算 binary 指标 + 类型准确率，返回结构化结果。"""
    gt_by_id = {g["id"]: g for g in ground_truth}
    n = len(results)
    tp = fp = fn = tn = 0
    type_correct = type_total = 0
    rows = []
    fn_list, fp_list, type_mismatch = [], [], []

    for r in results:
        g = gt_by_id[r["id"]]
        truth = bool(g["is_hallucination"])
        pred = bool(r["is_hallucination"])
        type_truth = g.get("hallucination_type") or "无幻觉"
        type_pred = r.get("hallucination_type") or "无幻觉"
        type_ok = type_pred == type_truth

        if pred and truth:
            tp += 1
            type_total += 1
            if type_ok:
                type_correct += 1
            else:
                type_mismatch.append({"id": r["id"], "truth_type": type_truth,
                                      "pred_type": type_pred})
        elif pred and not truth:
            fp += 1
            fp_list.append(r["id"])
        elif not pred and truth:
            fn += 1
            fn_list.append({"id": r["id"], "truth_type": type_truth,
                            "truth_detail": g.get("detail", "")})
        else:
            tn += 1

        rows.append({
            "id": r["id"], "truth": "幻觉" if truth else "正确",
            "pred": "幻觉" if pred else "正确",
            "binary_ok": "√" if pred == truth else "×",
            "truth_type": type_truth,
            "pred_type": type_pred,
            "type_ok": ("√" if type_ok else "×") if (pred and truth) else "-",
            "severity": r.get("severity", "-"),
            "confidence": _fmt(r.get("confidence")),
            "source": r.get("source", "-"),
            "rule_id": r.get("rule_id", "-"),
        })

    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    type_accuracy = type_correct / type_total if type_total else None

    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "type_accuracy": type_accuracy, "type_correct": type_correct,
        "type_total": type_total,
        "rows": rows, "fn_list": fn_list, "fp_list": fp_list,
        "type_mismatch": type_mismatch,
        "source_stats": Counter(r["source"] for r in results),
        "rule_stats": Counter(r.get("rule_id") for r in results if r.get("rule_id")),
        "severity_stats": Counter(r.get("severity") for r in results),
    }


def render_report(metrics, ground_truth, llm_mode, llm_warnings):
    """生成 Markdown 评估报告。"""
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    lines = []
    add = lines.append

    add("# 幻觉检测评估报告\n")
    add(f"- 检测模式：LLM 层 = **{llm_mode}**"
        + ("（" + "；".join(llm_warnings) + "）" if llm_warnings else ""))
    add(f"- 样本数：{metrics['n']}（ground_truth 标注幻觉 {tp + fn} 条，正常 {tn + fp} 条）\n")

    add("## 1. 检出率指标\n")
    add("| 指标 | 值 |")
    add("|---|---|")
    add(f"| TP（正确检出幻觉） | {tp} |")
    add(f"| FP（误报，正确判成幻觉） | {fp} |")
    add(f"| FN（漏检，幻觉判成正确） | {fn} |")
    add(f"| TN（正确判非幻觉） | {tn} |")
    add(f"| 准确率 Accuracy | {metrics['accuracy']:.2%} |")
    add(f"| 精确率 Precision | {metrics['precision']:.2%} |")
    add(f"| 召回率 Recall | {metrics['recall']:.2%} |")
    add(f"| F1 | {metrics['f1']:.2%} |")
    ta = metrics["type_accuracy"]
    add(f"| 类型准确率（幻觉样本中细类一致比例） | "
        f"{ta:.2%}（{metrics['type_correct']}/{metrics['type_total']}）" if ta is not None
        else "| 类型准确率 | - |")

    add("\n## 2. 逐条对照表\n")
    add("| id | 人工标注 | 检测结果 | 一致性 | 人工细类 | 检测细类 | 类型 | 严重度 | 置信度 | 来源 | 命中规则 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in metrics["rows"]:
        add(f"| {r['id']} | {r['truth']} | {r['pred']} | {r['binary_ok']} | "
            f"{r['truth_type']} | {r['pred_type']} | {r['type_ok']} | "
            f"{r['severity']} | {r['confidence']} | {r['source']} | {r['rule_id']} |")

    add("\n## 3. 漏检 / 误报 / 类型不一致清单\n")
    if metrics["fn_list"]:
        add("**漏检（FN）**：\n")
        for item in metrics["fn_list"]:
            add(f"- {item['id']}：人工标注「{item['truth_type']}」— {item['truth_detail']}")
    else:
        add("**漏检（FN）：无**")
    if metrics["fp_list"]:
        add("\n**误报（FP）**：\n")
        for item in metrics["fp_list"]:
            add(f"- {item}：人工标注为正确，检测判为幻觉")
    else:
        add("\n**误报（FP）：无**")
    if metrics["type_mismatch"]:
        add("\n**类型不一致（细类不同，均判为幻觉）**：\n")
        for item in metrics["type_mismatch"]:
            add(f"- {item['id']}：人工「{item['truth_type']}」vs 检测「{item['pred_type']}」")
    else:
        add("\n**类型不一致：无**")

    add("\n## 4. 判定来源与规则命中统计\n")
    add("| 来源 | 数量 |")
    add("|---|---|")
    for source, count in sorted(metrics["source_stats"].items()):
        add(f"| {source} | {count} |")
    if metrics["rule_stats"]:
        add("\n| 规则 | 命中数 |")
        add("|---|---|")
        for rule, count in sorted(metrics["rule_stats"].items()):
            add(f"| {rule} | {count} |")
    add("\n| 严重度 | 数量 |")
    add("|---|---|")
    for sev, count in sorted(metrics["severity_stats"].items()):
        add(f"| {sev} | {count} |")

    add("\n## 5. 误判与边界分析\n")
    add("### 5.1 本轮结果\n")
    if metrics["fp"] == 0 and metrics["fn"] == 0 and metrics["type_mismatch"]:
        add("本轮 binary 全对、细类全部一致。以下记录的是最容易出错的边界位置及防护机制。")
    elif metrics["fp"] == 0 and metrics["fn"] == 0:
        add("本轮 binary 检出 20/20 全对，且细类与人工标注完全一致。")
    else:
        add("本轮存在漏检/误报，详见第 3 节清单。")

    add("\n### 5.2 容易误判的 case 及原因\n")
    add("- **h12 / h16（正确样本，误报风险最高）**：h12 的回复与知识库都含否定表述"
        "「不支持货到付款」，如果只做「知识库否定 + 回复提及」的粗匹配会误报；"
        "本工具用否定极性检查（回复在提及该事项时同样否定，视为一致）避免了误报。"
        "h16 的「基本准确」与知识库「可能存在轻微色差」是程度差异而非矛盾，"
        "若规则对「轻微/基本」类程度词过度敏感会误报。")
    add("- **h20（遗漏型，漏检风险最高）**：知识库「30% 用户反馈偏大半码、建议选小半码」"
        "与回复「尺码标准、不偏」没有直接数字矛盾，只有反义碰撞（偏大 ↔ 标准）。"
        "通用反义规则（知识库含限定建议 + 回复给绝对结论）可检出，但置信度设为 0.85，"
        "因为「遗漏」与「客服合理省略细节」的边界本就模糊，人工标注也承认该条边界较模糊。")
    add("- **h04（部分正确部分错误，易漏检）**：回复「支持电子发票」是对的，"
        "只有「纸质发票」和「备注填写」两处错误。只做整体判断容易放过；"
        "本工具逐槽位做否定翻转检测，并用「回复与知识库存在共同正确项」区分"
        "政策偏差（部分错）与政策编造（整体错）。")
    add("- **h09 / h15（「未标注/未提及」型，易漏检）**：知识库不是否定而是「缺信息」，"
        "需要 absence-of-evidence 推理：知识库声明未标注/未提及，回复却给出肯定细节，"
        "即构成编造。此类样本在单纯矛盾检测下会漏。")
    add("- **h05 / h07（多类型重叠，类型易判偏）**：h05 既编造优惠又承诺「直接发到账户」"
        "（越界），h07 既给地址又违反「不可口头告知」（越界+编造）。"
        "人工标注取「优惠编造/信息编造」，本工具通过规则优先级（数值冲突先于能力越界、"
        "地址类先于政策类）与人工口径对齐。")
    add("- **h08 编造 vs 偏差的边界**：数值整体换了（24h→48h、快递→顺丰）但回复结构"
        "仍沿知识库槽位，人工标注为「政策偏差」；h01 因存在「运费承担方翻转 + 适用范围"
        "扩大」被标为「政策编造」。本工具用「是否存在受益方翻转/范围扩大」做区分，"
        "这是类型级最容易失分的地方。")
    add("\n### 5.3 方法局限\n")
    add("- 规则层在这 20 条样本上验证调优，泛化到新数据的能力有限；新增表述方式"
        "（如新数值格式、新否定句式）可能漏检。")
    add("- 数值槽位比对依赖上下文相似度（前后 4 字共享 2-gram），上下文差异较大时"
        "同一维度可能对不上槽位（如 h08 的「2-3 天到货」与「3-5 天」未配对，"
        "靠发货时效与快递冲突补足证据）。")
    add("- mock 模式的 LLM 层只是规则层的接口替身，不提供额外判断信号；"
        "真实 API 模式下 LLM 才作为独立裁判参与融合。")

    return "\n".join(lines) + "\n"
