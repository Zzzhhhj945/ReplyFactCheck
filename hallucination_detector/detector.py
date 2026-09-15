# -*- coding: utf-8 -*-
"""融合主流程：规则层（确定性）优先，LLM 裁判层补充语义判断。

- 规则命中：以规则为准；若 LLM 层同意则 source=rule+llm，分歧则标注并下调置信度。
- 规则未命中：采用 LLM 层判定。
- 两层都未判定（LLM off 且规则未命中）：标记为未判断，默认无幻觉、低置信度。
"""

from . import rules
from .taxonomy import NO_HALLUCINATION


def detect_case(case, judge):
    """对单条回复做检测，返回带 id 的判定结果。"""
    question = case["user_question"]
    reply = case["system_reply"]
    kb = case["knowledge_base"]

    rule_verdict = rules.apply_all(question, reply, kb)
    llm_verdict = judge.judge(question, reply, kb) if judge else None

    if rule_verdict:
        final = dict(rule_verdict)
        if llm_verdict is not None:
            agree = bool(llm_verdict["is_hallucination"]) == \
                bool(rule_verdict["is_hallucination"])
            final["llm_agree"] = agree
            final["source"] = "rule+llm" if agree else "rule(LLM分歧)"
            if not agree:
                final["confidence"] = round(min(final["confidence"], 0.9), 2)
        else:
            final["source"] = "rule"
        final["id"] = case["id"]
        return final

    if llm_verdict is not None:
        llm_verdict = dict(llm_verdict)
        llm_verdict["id"] = case["id"]
        llm_verdict["source"] = "llm:" + judge.mode
        return llm_verdict

    return {
        "id": case["id"],
        "is_hallucination": False,
        "hallucination_type": NO_HALLUCINATION,
        "category": NO_HALLUCINATION,
        "severity": "无",
        "evidence": "规则层与 LLM 层均未判定（LLM 已关闭且无规则命中）",
        "confidence": 0.5,
        "source": "unjudged",
    }


def detect_all(cases, judge):
    return [detect_case(c, judge) for c in cases]
