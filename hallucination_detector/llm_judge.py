# -*- coding: utf-8 -*-
"""LLM 裁判层：对规则层未覆盖的语义判断（偏差、遗漏、部分正确）做独立裁决。

两种模式（run.py --llm 可选）：
- auto（默认）：检测到环境变量中有 OpenAI 兼容的 API key 就走真实 API，否则降级 mock。
- real：强制真实 API，未配置 key 时报错退出。
- mock：完全离线。mock 不是真实模型，而是「用规则层的确定性判断模拟 LLM
  返回格式」的接口替身，用于离线复现与演示，README 中如实说明。
- off：跳过 LLM 层，仅规则层。
"""

import json
import os
import re
import urllib.error
import urllib.request

from . import rules
from .taxonomy import ALL_TYPES, NO_HALLUCINATION, category_of

PROVIDERS = (
    # (名称, key 环境变量, 默认 base_url, 默认模型)
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("moonshot", "MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("zhipu", "ZHIPU_API_KEY", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
)

SYSTEM_PROMPT = (
    "你是电商客服回复幻觉检测器。给定【知识库】、【用户问题】和【系统回复】，"
    "判断系统回复是否相对知识库产生了幻觉，并只输出一个 JSON 对象，不要输出任何其他文字。\n"
    "幻觉细类定义：\n"
    "- 参数编造：产品参数（蓝牙版本、接口、材质、保修期等）与知识库不符，或知识库未提供\n"
    "- 政策编造：退货/运费等政策被整体编造成对客户更有利的版本\n"
    "- 优惠编造：杜撰不存在的优惠券、折扣活动或优惠入口\n"
    "- 信息编造：杜撰地址、门店、品牌关系等知识库没有的信息\n"
    "- 能力越界：知识库声明未接入接口/无功能，回复却声称已查询/已修改/已升级\n"
    "- 安全误导：违背知识库健康安全风险提示，给出放心承诺\n"
    "- 政策偏差：部分正确部分错误，数值/渠道/时效与知识库不一致\n"
    "- 信息遗漏：遗漏知识库关键限定信息，导致结论失真\n"
    "- 无幻觉：回复与知识库一致或兼容（回复可省略细节，但不能与知识库矛盾）\n"
    "严重度：高/中/低/无。判断标准：影响维度（健康安全>交易资金>服务承诺>产品认知>营销体验）"
    "与偏离程度（完全杜撰>偏差>遗漏）共同决定。\n"
    "输出格式：{\"is_hallucination\": bool, \"hallucination_type\": \"细类\", "
    "\"severity\": \"高|中|低|无\", \"evidence\": \"引用知识库与回复的矛盾点\", "
    "\"confidence\": 0到1之间的小数}"
)

# few-shot 全部为合成示例，不取自这 20 条评测样本，避免测试集泄漏
FEW_SHOT = [
    {
        "kb": "产品参数：蓝牙4.2，延迟约100ms。",
        "question": "这个耳机蓝牙版本是多少？",
        "reply": "这款耳机是蓝牙5.3，延迟低至20ms。",
        "answer": {"is_hallucination": True, "hallucination_type": "参数编造",
                   "severity": "高", "evidence": "知识库为蓝牙4.2/延迟100ms，回复改为5.3/20ms",
                   "confidence": 0.98},
    },
    {
        "kb": "支付方式：微信支付、支付宝。不支持货到付款。",
        "question": "可以货到付款吗？",
        "reply": "目前不支持货到付款，支持微信和支付宝。",
        "answer": {"is_hallucination": False, "hallucination_type": "无幻觉",
                   "severity": "无", "evidence": "回复与知识库一致",
                   "confidence": 0.95},
    },
    {
        "kb": "成分：含酒精。注意事项：敏感肌慎用。",
        "question": "敏感肌能用吗？",
        "reply": "不含酒精，敏感肌可以放心使用。",
        "answer": {"is_hallucination": True, "hallucination_type": "安全误导",
                   "severity": "高", "evidence": "知识库提示敏感肌慎用，回复反向承诺放心使用",
                   "confidence": 0.97},
    },
    {
        "kb": "用户评价汇总：约40%用户反馈偏小一码，建议选大一码。",
        "question": "这双鞋偏大还是偏小？",
        "reply": "这款鞋尺码标准，按平时尺码选即可。",
        "answer": {"is_hallucination": True, "hallucination_type": "信息遗漏",
                   "severity": "低", "evidence": "知识库有偏小建议，回复遗漏并给出相反结论",
                   "confidence": 0.85},
    },
]


def available_provider():
    """返回 (名称, key, base_url, 模型) 或 None。"""
    for name, key_env, base, model in PROVIDERS:
        key = os.environ.get(key_env)
        if key:
            base = os.environ.get(f"{name.upper()}_BASE_URL") or base
            model = os.environ.get(f"{name.upper()}_MODEL") or model
            return name, key, base, model
    return None


def _parse_json(content):
    """从模型输出中提取 JSON 对象。"""
    if not content:
        return None
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LLMJudge:
    def __init__(self, mode="auto"):
        self.mode = "off"
        self.warnings = []
        if mode == "off":
            return
        if mode == "mock":
            self.mode = "mock"
            return
        provider = available_provider()
        if provider is None:
            if mode == "real":
                raise SystemExit(
                    "强制 real 模式但未检测到 API key。请设置 DEEPSEEK_API_KEY / "
                    "OPENAI_API_KEY / MOONSHOT_API_KEY / ZHIPU_API_KEY 之一。")
            self.mode = "mock"
            self.warnings.append("未检测到 LLM API key，LLM 层降级为 mock 模式")
            return
        self.mode = "real"
        self.provider, self.api_key, self.base_url, self.model = provider

    # ------------------------------------------------------------ 对外接口
    def judge(self, question, reply, kb):
        if self.mode == "off":
            return None
        if self.mode == "mock":
            return self._judge_mock(question, reply, kb)
        return self._judge_real(question, reply, kb)

    # ------------------------------------------------------------ mock 模式
    def _judge_mock(self, question, reply, kb):
        """mock = 规则层的确定性判断按 LLM 的返回格式封装，模拟语义裁判的输出。"""
        verdict = rules.apply_all(question, reply, kb)
        if verdict:
            return {
                "is_hallucination": True,
                "hallucination_type": verdict["hallucination_type"],
                "severity": verdict["severity"],
                "evidence": "(mock) " + verdict["evidence"],
                "confidence": 0.85,
            }
        return {
            "is_hallucination": False,
            "hallucination_type": NO_HALLUCINATION,
            "category": category_of(NO_HALLUCINATION),
            "severity": "无",
            "evidence": "未发现回复与知识库的矛盾点（mock 模式启发式判断）",
            "confidence": 0.8,
        }

    # ------------------------------------------------------------ 真实 API
    def _judge_real(self, question, reply, kb):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEW_SHOT:
            messages.append({"role": "user",
                             "content": f"知识库：{ex['kb']}\n用户问题：{ex['question']}\n系统回复：{ex['reply']}"})
            messages.append({"role": "assistant",
                             "content": json.dumps(ex["answer"], ensure_ascii=False)})
        messages.append({"role": "user",
                         "content": f"知识库：{kb}\n用户问题：{question}\n系统回复：{reply}"})
        try:
            content = self._call_chat(messages)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, ValueError) as exc:
            self.warnings.append(f"LLM API 调用失败（{type(exc).__name__}），本条降级为 mock")
            return self._judge_mock(question, reply, kb)
        obj = _parse_json(content)
        if obj is None:
            self.warnings.append("LLM 返回无法解析为 JSON，本条降级为 mock")
            return self._judge_mock(question, reply, kb)
        return self._normalize(obj)

    def _call_chat(self, messages):
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _normalize(obj):
        htype = obj.get("hallucination_type")
        if htype not in ALL_TYPES:
            htype = NO_HALLUCINATION
        is_h = bool(obj.get("is_hallucination", False))
        if not is_h:
            htype = NO_HALLUCINATION
        severity = obj.get("severity")
        if severity not in ("高", "中", "低", "无"):
            severity = "无" if htype == NO_HALLUCINATION else "中"
        conf = obj.get("confidence")
        try:
            conf = min(max(float(conf), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        return {
            "is_hallucination": is_h,
            "hallucination_type": htype,
            "category": category_of(htype),
            "severity": severity,
            "evidence": str(obj.get("evidence", ""))[:500],
            "confidence": conf,
        }
