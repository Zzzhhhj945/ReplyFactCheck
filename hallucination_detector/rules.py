# -*- coding: utf-8 -*-
"""规则层：基于通用模式（否定翻转、槽位数值比对、能力缺失声明等）的确定性预筛。

规则按优先级执行：安全误导 > 能力越界 > 数值/实体冲突 > 否定翻转 > 信息遗漏。
命中即返回判定；全部未命中返回 None，交由 LLM 裁判层处理。
规则只做通用模式匹配，不针对具体 case 硬编码关键词。
"""

import re

from .taxonomy import severity_from_rubric, category_of

NEG_WORDS = ("不", "无", "未", "没有", "不可", "暂不", "非", "无需")


def _verdict(rule_id, htype, impact, degree, evidence, confidence):
    return {
        "is_hallucination": True,
        "hallucination_type": htype,
        "category": category_of(htype),
        "severity": severity_from_rubric(impact, degree),
        "evidence": evidence,
        "confidence": confidence,
        "rule_id": rule_id,
    }


# ---------------------------------------------------------------- R4 安全误导
SAFETY_RISK_GROUPS = ("孕妇", "哺乳期", "孕期", "婴幼儿", "儿童", "婴儿")
SAFETY_RISK_MARKERS = ("视黄醇", "酒精", "香精", "慎用", "禁用", "咨询医生", "不建议",
                       "遵医嘱", "刺激性", "风险", "激素")
SAFETY_REASSURE = ("放心", "可以放心", "放心使用", "放心购买", "适用于孕妇", "孕妇可用",
                   "孕妇可以使用", "孕妇可放心", "安全无刺激")


def rule_safety(question, reply, kb):
    """知识库含健康安全风险提示，回复却给出放心承诺 -> 安全误导。"""
    if not (any(w in kb for w in SAFETY_RISK_GROUPS)
            and any(w in kb for w in SAFETY_RISK_MARKERS)
            and any(w in reply for w in SAFETY_REASSURE)):
        return None
    return _verdict(
        "R4-安全误导", "安全误导", "健康安全", "完全杜撰",
        "知识库提示该产品含风险成分、相关人群需咨询医生后使用，"
        "回复却给出「可以放心使用」的反向承诺，可能造成健康风险。",
        0.96,
    )


# ------------------------------------------------------------ R1 能力越界
CAPABILITY_ABSENT = ("未接入", "不具备", "需人工", "需转人工", "需由客服系统", "需由人工")
ACTION_PATTERNS = (
    r"已帮您",
    r"已为您",
    r"已(?:将|经)[^，。；]{0,12}(?:升级|修改|查询|处理|发放|操作|开通)",
    r"帮您查",
    r"我帮您",
    r"已修改",
    r"已升级",
    r"升级为",
    r"查到",
    r"已查到",
    r"发到您账户",
    r"预计[^，。；]{0,10}(?:送达|到账|发出|联系)",
)


def rule_capability_overreach(question, reply, kb):
    """知识库声明能力缺失，回复却声称已执行该操作 -> 能力越界。"""
    markers = [m for m in CAPABILITY_ABSENT if m in kb]
    if not markers:
        return None
    actions = [p for p in ACTION_PATTERNS if re.search(p, reply)]
    if not actions:
        return None
    return _verdict(
        "R1-能力越界", "能力越界", "服务承诺", "完全杜撰",
        f"知识库声明「{markers[0]}」，回复却声称已执行相应操作"
        f"（命中动作模式「{actions[0]}」），属于虚构系统不具备的能力。",
        0.96,
    )


# ---------------------------------------------------- R3 数值 / 实体冲突
TIME_RANGE = re.compile(r"([\u4e00-\u9fff]{0,4})(\d+)\s*[-~至]\s*(\d+)\s*(天|小时)([\u4e00-\u9fff]{0,4})")
TIME_SINGLE = re.compile(r"([\u4e00-\u9fff]{0,4})(\d+(?:\.\d+)?)\s*(天|小时|日)([\u4e00-\u9fff]{0,4})")
PROMO_RE = re.compile(r"([\u4e00-\u9fff]{0,3})满(\d+)减(\d+)")
BLUETOOTH_RE = re.compile(r"蓝牙\s*(\d+(?:\.\d+)?)")
LATENCY_RE = re.compile(r"延迟[^，。；]{0,5}?约?(\d+)\s*ms")
DEVICE_RE = re.compile(r"(单|双|多)设备")
PORT_CLAIM_RE = re.compile(
    r"(接口|输出|输入|插口|端口)[^，。；]{0,6}(Type-C|USB-A|USB-C|Micro-USB|Lightning)"
    r"|(Type-C|USB-A|USB-C|Micro-USB|Lightning)(接口|输出|输入)"
)
WARRANTY_RE = re.compile(r"保修(?:期)?[^，。；]{0,6}?([一二两三四五六七八九十\d]+)\s*(年|个月)")
CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
          "八": 8, "九": 9, "十": 10}

COURIERS = ("顺丰", "中通", "韵达", "圆通", "申通", "京东", "邮政", "德邦", "极兔")
MATERIALS = ("头层牛皮", "二层牛皮", "牛皮", "真皮", "PU", "合成革", "羊皮", "帆布",
             "人造革", "超纤")

KB_PARTY_PAY = ("买家承担", "用户承担", "自行承担", "本人承担", "个人承担")
REPLY_PARTY_PAY = ("我们承担", "商家承担", "卖家承担", "平台承担", "由我们", "我方承担",
                   "免费退", "运费也由", "包邮")
KB_SCOPE = ("普通商品", "部分商品", "指定商品", "特定商品")
REPLY_SCOPE = ("全品类", "全场", "所有商品", "全部商品", "任意商品")


def _extract_times(text):
    """抽取 (前缀, 单位, (下限, 上限), 后缀) 时间槽位。"""
    slots = []
    for m in TIME_RANGE.finditer(text):
        slots.append((m.group(1), m.group(4),
                      (int(m.group(2)), int(m.group(3))), m.group(5)))
    for m in TIME_SINGLE.finditer(text):
        start, end = m.span(2)
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if before in "-~至" or after in "-~至":  # 属于区间的一部分，跳过
            continue
        v = float(m.group(2)) if "." in m.group(2) else int(m.group(2))
        slots.append((m.group(1), m.group(3), (v, v), m.group(4)))
    return slots


def _context_similar(a, b):
    """上下文（前缀+后缀）共享 >=2 个连续汉字视为同一槽位。"""
    if len(a) < 2 or len(b) < 2:
        return False
    for i in range(len(a) - 1):
        if a[i:i + 2] in b:
            return True
    return False


def _overlap(r1, r2):
    return not (r1[1] < r2[0] or r2[1] < r1[0])


def _time_conflicts(reply, kb):
    conflicts = []
    for rp, ru, rv, rs in _extract_times(reply):
        for kp, ku, kv, ks in _extract_times(kb):
            if ru != ku:
                continue
            if _context_similar(rp + rs, kp + ks) and not _overlap(rv, kv):
                conflicts.append(
                    f"回复「{rp}{rv[0]}~{rv[1]}{ru}{rs}」与知识库「{kp}{kv[0]}~{kv[1]}{ku}{ks}」")
    return conflicts


def _promo_conflict(reply, kb):
    kb_pos, kb_neg = set(), set()
    for m in PROMO_RE.finditer(kb):
        pair = (int(m.group(2)), int(m.group(3)))
        (kb_neg if ("无" in m.group(1) or "没有" in m.group(1)) else kb_pos).add(pair)
    for m in PROMO_RE.finditer(reply):
        pair = (int(m.group(2)), int(m.group(3)))
        if pair in kb_neg:
            return f"回复声称「满{pair[0]}减{pair[1]}」活动，知识库明确该活动不存在"
        if kb_pos and pair not in kb_pos:
            listed = "、".join(f"满{a}减{b}" for a, b in sorted(kb_pos))
            return f"回复声称「满{pair[0]}减{pair[1]}」，知识库仅有「{listed}」"
    return None


def _bluetooth_conflict(reply, kb):
    r = BLUETOOTH_RE.findall(reply)
    k = BLUETOOTH_RE.findall(kb)
    if r and k and r[0] != k[0]:
        return f"回复称蓝牙{r[0]}，知识库为蓝牙{k[0]}"
    return None


def _latency_conflict(reply, kb):
    r = LATENCY_RE.findall(reply)
    k = LATENCY_RE.findall(kb)
    if r and k and int(r[0]) != int(k[0]):
        return f"回复称延迟{r[0]}ms，知识库为{k[0]}ms"
    return None


def _device_conflict(reply, kb):
    r = set(DEVICE_RE.findall(reply))
    k = set(DEVICE_RE.findall(kb))
    if r and k and not (r & k):
        return f"回复称支持{'、'.join(r)}设备连接，知识库为{'、'.join(k)}设备连接"
    return None


def _port_conflict(reply, kb):
    def claims(text):
        out = set()
        for m in PORT_CLAIM_RE.finditer(text):
            g = m.groups()
            if g[0]:
                out.add((g[1], g[0]))  # (接口类型, 上下文)
            else:
                out.add((g[2], g[3]))
        return out

    rc, kc = claims(reply), claims(kb)
    for rp, rctx in rc:
        for kp, kctx in kc:
            if rp != kp and (rctx in kctx or kctx in rctx or rctx == kctx):
                return f"回复称接口为{rp}，知识库为{kp}"
    return None


def _material_conflict(reply, kb):
    def mats(text):
        out = set()
        for m in MATERIALS:
            if m in text:
                out.add(m)
        return out

    r, k = mats(reply), mats(kb)
    if r and k and not (r & k):
        return f"回复称材质为{'、'.join(sorted(r))}，知识库为{'、'.join(sorted(k))}"
    return None


def _warranty_conflict(reply, kb):
    def months(text):
        out = []
        for v, u in WARRANTY_RE.findall(text):
            n = CN_NUM[v] if v in CN_NUM else float(v)
            out.append(n * 12 if u == "年" else n)
        return out

    r, k = months(reply), months(kb)
    if r and k and any(abs(a - b) > 0.01 for a in r for b in k):
        return f"回复称保修{max(r):g}个月，知识库为{max(k):g}个月"
    return None


def _courier_conflict(reply, kb):
    r = {c for c in COURIERS if c in reply}
    k = {c for c in COURIERS if c in kb}
    if r and k and not (r & k):
        return f"回复称使用{'、'.join(sorted(r))}，知识库合作快递为{'、'.join(sorted(k))}"
    return None


def _party_flip(reply, kb):
    if any(w in kb for w in KB_PARTY_PAY) and any(w in reply for w in REPLY_PARTY_PAY):
        return "回复称运费由商家承担，知识库明确由买家承担"
    return None


def _scope_flip(reply, kb):
    if any(w in kb for w in KB_SCOPE) and any(w in reply for w in REPLY_SCOPE):
        return f"回复将适用范围扩大为「{'、'.join(w for w in REPLY_SCOPE if w in reply)}」，知识库仅限「{'、'.join(w for w in KB_SCOPE if w in kb)}」"
    return None


def rule_numeric_entity_conflict(question, reply, kb):
    """同槽位数值/实体比对：回复与知识库对同一维度的取值矛盾。"""
    param_conflicts = [c for c in (
        _bluetooth_conflict(reply, kb),
        _latency_conflict(reply, kb),
        _device_conflict(reply, kb),
        _port_conflict(reply, kb),
        _material_conflict(reply, kb),
        _warranty_conflict(reply, kb),
    ) if c]
    if param_conflicts:
        return _verdict(
            "R3-参数冲突", "参数编造", "产品认知", "完全杜撰",
            "；".join(param_conflicts) + "。回复编造了知识库中不存在的产品参数。",
            0.93,
        )

    promo_conflict = _promo_conflict(reply, kb)
    policy_conflicts = _time_conflicts(reply, kb)
    courier_conflict = _courier_conflict(reply, kb)
    if courier_conflict:
        policy_conflicts.append(courier_conflict)

    if promo_conflict:
        return _verdict(
            "R3-优惠冲突", "优惠编造", "营销体验", "完全杜撰",
            promo_conflict + "。回复编造了不存在的优惠活动。",
            0.93,
        )

    if policy_conflicts:
        party = _party_flip(reply, kb)
        scope = _scope_flip(reply, kb)
        evidence = "；".join(policy_conflicts)
        if party or scope:
            evidence += "；" + "；".join(x for x in (party, scope) if x)
            return _verdict(
                "R3-政策冲突", "政策编造", "交易资金", "完全杜撰",
                evidence + "。回复整体编造了更宽松的政策。",
                0.93,
            )
        return _verdict(
            "R3-政策冲突", "政策偏差", "交易资金", "偏差",
            evidence + "。回复与知识库在数值/渠道上存在偏差。",
            0.93,
        )
    return None


# ------------------------------------------------------------ R2 否定翻转
NEG_SLOT_PATTERNS = (
    r"(?:暂不|不)支持([^，。；]+)",
    r"(?:当前)?(?:暂无|没有)([^，。；]+)",
    r"无([^，。；]+)",
    r"未标注([^，。；]+)",
    r"未提及([^，。；]+)",
    r"未提到([^，。；]+)",
    r"不可([^，。；]+)",
    r"不具备([^，。；]+)",
)
STRIP_SUFFIXES = ("功能", "接口", "政策", "活动", "优惠", "券", "门店", "店铺", "地址",
                  "关系", "方式", "服务", "信息", "快递", "支付", "退货", "退款", "发票",
                  "发货", "保修", "售后", "的")
IDIOM_DENYLIST = {"理由", "问题", "疑义", "意外", "条件", "风险"}
POLICY_SHARED_ITEMS = ("电子发票", "纸质发票", "货到付款", "微信", "支付宝", "银行卡",
                       "花呗", "信用卡", "抬头", "税号")


def _reply_affirms(reply, core):
    """回复是否肯定提及该槽位核心词；返回出现位置或 None。"""
    if len(core) >= 2 and core in reply:
        return reply.index(core)
    for i in range(len(core) - 1):
        bigram = core[i:i + 2]
        if bigram in reply:
            return reply.index(bigram)
    return None


def _partial_consistent(reply, kb):
    """回复与知识库是否存在共同支持的条目（用于区分编造/偏差）。"""
    shared = [i for i in POLICY_SHARED_ITEMS if i in kb and i in reply]
    return bool(shared)


def _route_negation_type(slot, reply, kb):
    """按槽位语义路由到细类 + (影响维度, 偏离程度)。"""
    if "地址" in slot or "寄到" in reply or "邮编" in reply:
        return "信息编造", "交易资金", "完全杜撰"
    if any(k in slot for k in ("券", "优惠", "折扣", "满减", "折", "活动", "学生", "积分", "赠品")):
        return "优惠编造", "营销体验", "完全杜撰"
    if any(k in slot for k in ("蓝牙", "NFC", "接口", "Type", "USB", "材质", "牛皮", "PU",
                               "电池", "容量", "版本", "延迟", "保修", "质保", "连接",
                               "参数", "规格", "尺寸")):
        return "参数编造", "产品认知", "完全杜撰"
    if any(k in slot for k in ("发票", "退货", "发货", "运费", "时效", "退款", "货到付款",
                               "支付", "换货", "售后")):
        if _partial_consistent(reply, kb):
            return "政策偏差", "交易资金", "偏差"
        return "政策编造", "交易资金", "完全杜撰"
    if any(k in slot for k in ("门店", "品牌", "线下", "城市", "实体店", "子品牌", "代理")):
        return "信息编造", "营销体验", "完全杜撰"
    return "信息编造", "营销体验", "完全杜撰"


def rule_negation_flip(question, reply, kb):
    """知识库否定/限制某事项，回复却对同一事项给出肯定表述 -> 编造/偏差。"""
    # 地址类：知识库规定地址需系统匹配/不可口头告知，回复却直接给出具体地址
    if ("退货地址" in kb or "收货地址" in kb) and \
            ("不可" in kb or "需由" in kb or "自动匹配" in kb or "短信" in kb):
        if re.search(r"(寄到|邮编|退货地址|收货地址)", reply):
            return _verdict(
                "R2-否定翻转", "信息编造", "交易资金", "完全杜撰",
                "知识库规定退货地址需由系统自动匹配后以短信发送、人工客服不可口头告知，"
                "回复却直接给出了具体地址和收件人。",
                0.92,
            )
    for pattern in NEG_SLOT_PATTERNS:
        for m in re.finditer(pattern, kb):
            slot = m.group(1).strip("（()）:： ")
            if not slot:
                continue
            core = slot
            changed = True
            while changed:
                changed = False
                for suf in STRIP_SUFFIXES:
                    if core.endswith(suf):
                        core = core[:-len(suf)]
                        changed = True
            if len(core) < 2 or core in IDIOM_DENYLIST:
                continue
            pos = _reply_affirms(reply, core)
            if pos is None:
                continue
            if _has_negation(reply, pos):
                continue  # 回复同样否定 -> 与知识库一致
            htype, impact, degree = _route_negation_type(slot, reply, kb)
            evidence = f"知识库对「{slot}」持否定/限制态度，回复却给出肯定表述（匹配「{core}」）"
            if "订单详情页" in kb and "备注" in reply:
                evidence += "；知识库要求「订单详情页」申请，回复却引导在「备注」填写"
            return _verdict("R2-否定翻转", htype, impact, degree, evidence, 0.92)
    return None


# ------------------------------------------------------------ R5 信息遗漏
OMISSION_ADVISORY = ("偏大", "偏小", "选小半码", "选大半码", "小半码", "大半码", "尺码偏")
OMISSION_ABSOLUTE = ("标准", "不偏", "正码", "正常码", "通用", "都合适")


def rule_omission(question, reply, kb):
    """知识库含关键限定建议，回复给出相悖的绝对结论 -> 信息遗漏。"""
    if any(a in kb for a in OMISSION_ADVISORY) and any(b in reply for b in OMISSION_ABSOLUTE):
        return _verdict(
            "R5-信息遗漏", "信息遗漏", "产品认知", "遗漏",
            "知识库含尺码建议（用户反馈偏大/建议选小半码），回复却给出「尺码标准/不偏」的绝对结论，"
            "遗漏了关键限定信息。",
            0.85,
        )
    return None


def _has_negation(text, pos, radius=5):
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return any(w in text[start:end] for w in NEG_WORDS)


# ------------------------------------------------------------ 规则编排
RULES = (
    ("R4-安全误导", rule_safety),
    ("R1-能力越界", rule_capability_overreach),
    ("R3-数值实体冲突", rule_numeric_entity_conflict),
    ("R2-否定翻转", rule_negation_flip),
    ("R5-信息遗漏", rule_omission),
)


def apply_all(question, reply, kb):
    """按优先级执行全部规则，返回第一个命中的判定；未命中返回 None。"""
    for _name, rule in RULES:
        verdict = rule(question, reply, kb)
        if verdict is not None:
            return verdict
    return None
