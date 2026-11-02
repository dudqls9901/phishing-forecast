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
문자·통화 문구는 절대 작성하지 말고, 아래 JSON 형식으로만 답하라.
{{"scores": {{"<시나리오키>": 점수, ...}}, "reasons": {{"<시나리오키>": "한 문장 이유", ...}}}}

[사용자 일정]
{events}

[시나리오 구조]
{scenarios}"""


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


def llm_scores(events, timeout=8.0):
    """반환: (점수 dict, 이유 dict, 출처 'llm'|'rule')"""
    events = [e.strip() for e in events if e.strip()]
    if not events:
        return {s: 0.0 for s in S.SCENARIOS}, {s: "등록된 일정 없음" for s in S.SCENARIOS}, "rule"
    key = tuple(sorted(events))
    if key in _cache:
        return _cache[key]

    try:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("API 키 없음")
        sc_desc = "\n".join(
            f"- {k}: {v['name']} / 설득 원리: {', '.join(v['principles'])}" for k, v in S.SCENARIOS.items())
        client = anthropic.Anthropic(timeout=timeout)
        msg = client.messages.create(
            model=MODEL, max_tokens=600,
            messages=[{"role": "user", "content": PROMPT.format(
                events="\n".join(f"- {e}" for e in events), scenarios=sc_desc)}])
        text = msg.content[0].text
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        scores = {s: float(min(1, max(0, data["scores"].get(s, 0)))) for s in S.SCENARIOS}
        result = (scores, data.get("reasons", {}), "llm")
    except Exception as e:                     # 키 없음·네트워크·형식 오류 모두 규칙으로 대체
        print("LLM 대체 사유 :", type(e).__name__, e, flush=True)
        scores, keys = rule_scores(events)
        reasons = rule_reasons(events)
        reasons["_error"] = f"{type(e).__name__}: {e}"      # 화면에는 표시하지 않음 (디버깅용)
        reasons["_matched"] = keys
        result = (scores, reasons, "rule")
    _cache[key] = result
    return result


if __name__ == "__main__":
    for evs in (["다음 주 화요일 쿠팡 택배 도착 예정"], ["전세 계약 앞두고 대출 알아보는 중", "주말 이사"]):
        s, r, src = llm_scores(evs)
        print(evs, "→", src, {k: round(v, 2) for k, v in s.items()})
        if "_error" in r:
            print("    규칙 대체:", r["_error"])
