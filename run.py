# -*- coding: utf-8 -*-
"""入口：python run.py [--llm auto|real|mock|off]

流程：加载数据 -> 规则层 + LLM 裁判层融合检测 20 条回复 ->
写入 output/results.json -> 与 ground_truth 对齐评估 -> 写入 output/evaluation_report.md。
"""

import argparse
import json
import os
import sys

from hallucination_detector import detector, evaluate
from hallucination_detector.llm_judge import LLMJudge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="客服回复幻觉检测工具")
    parser.add_argument("--llm", choices=("auto", "real", "mock", "off"), default="auto",
                        help="LLM 裁判层模式（默认 auto：有 key 走真实 API，否则 mock）")
    parser.add_argument("--replies", default=os.path.join(BASE_DIR, "task4_replies.json"))
    parser.add_argument("--ground-truth", default=os.path.join(BASE_DIR, "task4_ground_truth.json"))
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "output"))
    args = parser.parse_args()

    cases = load_json(args.replies)
    ground_truth = load_json(args.ground_truth)

    judge = LLMJudge(mode=args.llm)
    print(f"[mode] LLM 裁判层: {judge.mode}")
    for w in judge.warnings:
        print(f"[warn] {w}")

    results = detector.detect_all(cases, judge)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics = evaluate.evaluate(results, ground_truth)
    report = evaluate.render_report(metrics, ground_truth, judge.mode, judge.warnings)
    report_path = os.path.join(args.out_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 控制台摘要
    print("\n" + "=" * 72)
    print("检测结果（人工标注 / 检测 => 一致性  细类）")
    print("=" * 72)
    for r in metrics["rows"]:
        mark = r["binary_ok"]
        print(f"{r['id']}  {r['truth']:>2}/{r['pred']:<2} {mark}  "
              f"{r['truth_type']:<5} -> {r['pred_type']:<5} "
              f"[{r['type_ok']}] sev={r['severity']} conf={r['confidence']} {r['source']}")
    print("-" * 72)
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"准确率={metrics['accuracy']:.2%}  精确率={metrics['precision']:.2%}  "
          f"召回率={metrics['recall']:.2%}  F1={metrics['f1']:.2%}")
    ta = metrics["type_accuracy"]
    print(f"类型准确率={ta:.2%}（{metrics['type_correct']}/{metrics['type_total']}）"
          if ta is not None else "类型准确率=-")
    print(f"结果文件: {results_path}")
    print(f"评估报告: {report_path}")

    return 0 if metrics["fn"] == 0 and metrics["fp"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
