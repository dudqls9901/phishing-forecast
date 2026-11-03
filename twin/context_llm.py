"""
맥락층: 사용자가 등록한 일정(자유 문장)과 공격 시나리오 구조가 얼마나 맞물리는지 0~1 점수로 판단한다.

- LLM은 '맞물림 정도' 판단에만 쓴다. 속을 확률은 통계 모델이 계산한다.
- LLM에게는 시나리오의 구조(유형·설득 원리·전제)만 주고, 문자·통화 대본은 만들게 하지 않는다.
- 시제품은 가상 사용자의 가상 일정만 사용한다. 목표 서비스에서는 기기 내 소형 모델로 대체.
- API 키가 없거나 호출이 실패하면 키워드 규칙으로 자동 대체 (시연 현장 네트워크 문제 대비).
"""
import json
import os
import re
import time

import scenarios as S

MODEL = os.environ.get("TWIN_LLM_MODEL", "claude-haiku-5-5")
_cache = {}

KEYWORDS = {
    "parcel": ["택배", "배송", "직구", "주문", "반품"],
    "loan": ["대출", "전세", "대환", "신용", "주택담보", "자금"],
    "family_trip": ["가족 여행", "여행", "해외", "출국"],
    "tax": ["연말정산", "세금", "환급", "종합소득세", "건강보험"],
    "moving": ["이사", "전입", "부동산"],
    "breach": ["유출", "해킹", "개인정보 사고"],
}

PROMPT = """너는 보안 앱의 맥락 판단 모듈이다. 사용자가 등록한 요즘 일정과, 사기 시나리오의 '구조'가 주어진다.
각 시나리오의 전제가 이 사용자에게 얼마나 그럴듯하게 들릴지를 0~1 점수로 판단하라.
- 1.0: 시나리오 전제가 사용자 일정과 직접 맞물림 (예: 택배를 기다리는 사람에게 배송 보류 안내)
- 0.5: 간접적으로 관련 있음
- 0.0: 관련 없음
또한 일정에 해당하는 맥락 키를 아래 [맥락 키]에서 모두 골라 contexts에 넣어라 (해당 없으면 빈 목록).
문자·통화 문구는 절대 작성하지 말고, 아래 JSON 형식으로만 답하라.
{{"scores": {{"<시나리오키>": 점수, ...}}, "reasons": {{"<시나리오키>": "한 문장 이유", ...}}, "contexts": ["<맥락키>", ...]}}

[사용자 일정]
{events}

[시나리오 구조]
{scenarios}

[맥락 키]
{contexts}"""

FAIL_TTL = 60          # 실패 결과는 60초만 기억 (일시적 네트워크 오류가 계속 남지 않도록)
_fail_cache = {}


def call_llm(prompt, timeout=20.0, max_tokens=2000):
    """LLM 호출 -> 응답 속 JSON(dict). 키 없음·네트워크·형식 오류는 예외로 올린다."""
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("API 키 없음")
    client = anthropic.Anthropic(timeout=timeout)
    msg = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def rule_scores(events):
    """키워드 규칙: 자유 문장 일정 -> 맥락 키 -> 규칙표 점수."""
    keys = {k for k, words in KEYWORDS.items() for e in events for w in words if w in e}
    return ({s: S.context_match(s, keys) for s in S.SCENARIOS}, sorted(keys))


def rule_reasons(events):
    """규칙 판단의 이유: 어떤 일정 문장이 어떤 맥락으로 읽혔는지 한 문장으로."""
    hits = {}                                   # 맥락 키 -> 처음 걸린 일정 문장
    for e in events:
        for k, words in KEYWORDS.items():
            if k not in hits and any(w in e for w in words):
                hits[k] = e
    out = {}
    for s, sc in S.SCENARIOS.items():
        best = max(((sc["match"].get(k, 0.0), k) for k in hits), default=(0.0, None))
        if best[0] > 0:
            k = best[1]
            how = "직접 맞물림" if best[0] >= 1 else "간접 관련"
            out[s] = f"'{hits[k]}' → {S.CONTEXTS[k]} 맥락과 {how}"
        else:
            out[s] = "관련 일정 없음"
    return out


def llm_scores(events, timeout=20.0):
    """반환: (점수 dict, 이유 dict, 출처 'llm'|'rule', 맥락 키 목록)"""
    events = [e.strip() for e in events if e.strip()]
    if not events:
        return ({s: 0.0 for s in S.SCENARIOS}, {s: "등록된 일정 없음" for s in S.SCENARIOS}, "rule", [])
    key = tuple(sorted(events))
    if key in _cache:
        return _cache[key]
    if key in _fail_cache and time.time() - _fail_cache[key][0] < FAIL_TTL:
        return _fail_cache[key][1]

    try:
        sc_desc = "\n".join(
            f"- {k}: {v['name']} / 설득 원리: {', '.join(v['principles'])}" for k, v in S.SCENARIOS.items())
        ctx_desc = "\n".join(f"- {k}: {v}" for k, v in S.CONTEXTS.items())
        data = call_llm(PROMPT.format(events="\n".join(f"- {e}" for e in events),
                                      scenarios=sc_desc, contexts=ctx_desc), timeout)
        scores = {s: float(min(1, max(0, data["scores"].get(s, 0)))) for s in S.SCENARIOS}
        contexts = sorted(c for c in data.get("contexts", []) if c in S.CONTEXTS)
        result = (scores, data.get("reasons", {}), "llm", contexts)
        _cache[key] = result                   # 성공한 결과만 계속 기억
    except Exception as e:                     # 키 없음·네트워크·형식 오류 모두 규칙으로 대체
        print("LLM 대체 사유 :", type(e).__name__, e, flush=True)
        scores, keys = rule_scores(events)
        reasons = rule_reasons(events)
        reasons["_error"] = f"{type(e).__name__}: {e}"      # 화면에는 표시하지 않음 (디버깅용)
        result = (scores, reasons, "rule", keys)
        _fail_cache[key] = (time.time(), result)
    return result


# ---------------------------------------------------------------- AI 비서 메모 -> 트윈 초안
PROFILE_PROMPT = """너는 보안 앱의 '디지털 트윈 초기 설정' 모듈이다. 사용자의 AI 비서가 기억하고 있는 메모가 주어진다.
1) 아래 설문 항목마다, 메모에 근거가 있을 때만 값을 추정하라. 근거가 없으면 null로 둔다. 추측으로 채우지 마라.
2) 메모에서 '요즘 진행 중이거나 곧 있을 일'을 한 줄짜리 문장으로 최대 5개 뽑아라. 이미 끝난 일은 제외한다.
건강·종교·정치 성향 등 민감한 내용은 추정하지 말고, 문자·통화 문구는 절대 작성하지 마라.
아래 JSON 형식으로만 답하라.
{{"survey": {{"<항목키>": 값 또는 null, ...}}, "reasons": {{"<항목키>": "메모 속 근거 한 문장", ...}}, "events": ["...", ...]}}

[설문 항목] (키: 설명, 범위)
{items}

[AI 비서 메모]
{memo}"""

PROFILE_ITEMS = {   # 키: (설명, 최소, 최대)
    "age": ("나이", 18, 75),
    "job_experience": ("일한 경력(년)", 0, 40),
    "cs_knowledge": ("컴퓨터·스마트폰 지식, 0=전혀 없음 5=전문가", 0, 5),
    "education": ("학력, 1=고졸 이하 2=전문대·재학 3=대졸 4=대학원", 1, 4),
    "self_confidence": ("피싱을 구별할 수 있다는 자신감, 0~5", 0, 5),
    "authority_suscept": ("기관·윗사람 요청에 따르는 정도, 0~5", 0, 5),
    "agreeableness": ("친화성(남을 잘 믿고 협조적), 0~5", 0, 5),
    "prior_attack": ("피싱·스미싱 피해 경험, 0=없음 1=있음", 0, 1),
    "consistency_susc": ("한번 한 약속·결정을 끝까지 지키려는 성향, 0~5", 0, 5),
    "anti_phish_course": ("피싱 예방 교육 수강, 0=없음 1=있음", 0, 1),
    "emotional_stab": ("정서 안정성, 0~5", 0, 5),
    "impulsivity": ("충동성(확인 전에 먼저 행동), 0~5", 0, 5),
}


def llm_profile(memo, timeout=30.0):
    """반환: {"survey": {키: 정수}, "reasons": {키: 문장}, "events": [문장]} — 근거 없는 항목은 survey에서 빠진다."""
    items = "\n".join(f"- {k}: {d} ({lo}~{hi})" for k, (d, lo, hi) in PROFILE_ITEMS.items())
    data = call_llm(PROFILE_PROMPT.format(items=items, memo=memo.strip()), timeout, max_tokens=3000)
    survey = {}
    for k, (_, lo, hi) in PROFILE_ITEMS.items():
        v = (data.get("survey") or {}).get(k)
        if isinstance(v, (int, float)):
            survey[k] = int(round(min(hi, max(lo, v))))
    events = [str(e).strip() for e in (data.get("events") or []) if str(e).strip()][:5]
    return {"survey": survey, "reasons": data.get("reasons") or {}, "events": events}


if __name__ == "__main__":
    for evs in (["다음 주 화요일 쿠팡 택배 도착 예정"], ["전세 계약 앞두고 대출 알아보는 중", "주말 이사"]):
        s, r, src, ctx = llm_scores(evs)
        print(evs, "→", src, {k: round(v, 2) for k, v in s.items()})
        if "_error" in r:
            print("    규칙 대체:", r["_error"])
