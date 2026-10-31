"""
공격 시나리오와 맥락 매칭 규칙 (구조 정보만, 실제 대본 없음)

- 각 시나리오는 유형 / 설득 원리 / 단계 구조만 기술한다.
- 상태 효과는 '결정 단계' 1곳에만 적용한다.
  근거 연구(시간 압박·스트레스 등)는 모두 메시지 1건에 대한 1회 판단을 측정했으므로,
  같은 효과를 모든 단계에 반복 적용하면 근거보다 효과가 부풀려진다.
- 단계별 통과 확률의 기본값(stage_offset)은 설계 가정이며, 결과는 '평소 대비 배수'로만 보고하므로
  절대 수준보다 상대 변화에 주로 쓰인다.
"""

CONTEXTS = {
    "parcel":     "택배 대기",
    "loan":       "대출 준비",
    "family_trip": "가족 여행",
    "tax":        "연말정산",
    "moving":     "이사",
    "breach":     "대형 유출 사고 보도",
}

# 맥락 중 '속았을 때 잃을 것이 큰 시기' (경고 민감도 상향에 사용)
HIGH_STAKES_CONTEXTS = {"loan", "moving"}

SCENARIOS = {
    "gov_impersonation": {
        "name": "기관 사칭 전화",
        "principles": ["권위", "긴급성"],
        "stages": ["접근", "신뢰 형성", "압박", "송금 요구"],
        "decision_stage": "압박",
        "match": {"breach": 1.0, "tax": 0.3, "loan": 0.3},
    },
    "loan_fraud": {
        "name": "저금리 대환대출 안내",
        "principles": ["호혜", "희소성"],
        "stages": ["접근", "신뢰 형성", "압박", "송금 요구"],
        "decision_stage": "압박",
        "match": {"loan": 1.0, "moving": 0.5},
    },
    "family_impersonation": {
        "name": "가족·지인 사칭 메신저",
        "principles": ["호감", "긴급성"],
        "stages": ["접근", "압박", "송금 요구"],
        "decision_stage": "압박",
        "match": {"family_trip": 1.0},
    },
    "parcel_smishing": {
        "name": "택배·공공기관 스미싱",
        "principles": ["일관성"],
        "stages": ["링크 클릭"],
        "decision_stage": "링크 클릭",
        "match": {"parcel": 1.0, "tax": 0.7, "moving": 0.6},
    },
}

# 결정 단계가 아닌 단계의 통과 logit 보정 (설계 가정):
# 한 번 대화에 응한 사람은 다음 단계로 넘어갈 가능성이 높다고 가정
NON_DECISION_STAGE_OFFSET = 1.0

# 맥락 일치 효과 (설계 가정): 완전히 맞물리는 맥락일 때의 OR 범위
# 정성 근거: 업무 맥락과 공격 전제가 맞을수록 클릭이 늘어난다는 현장 연구,
#           Zhuo et al. (2024)의 '업무 관련 피싱' 결과. 수치 근거는 확인한 범위에서 없음.
CONTEXT_OR_LOW, CONTEXT_OR_HIGH = 1.5, 3.0


def context_match(scenario_key, active_contexts):
    """규칙 기반 맥락 일치 점수 0~1 (가장 잘 맞는 맥락 기준). LLM 판단으로 교체 가능한 자리."""
    table = SCENARIOS[scenario_key]["match"]
    return max((table.get(c, 0.0) for c in active_contexts), default=0.0)
