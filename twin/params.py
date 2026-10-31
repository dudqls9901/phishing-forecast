"""
근거 파라미터 모음 (v1, 2026-10-28)

모든 값에는 출처와 근거 강도를 함께 적는다.
- "출처값"     : 논문에 보고된 수치를 그대로 사용
- "환산값"     : 논문 수치를 다른 척도로 변환 (변환 방법 명시)
- "설계 가정"  : 수치 근거가 없어 범위로 둔 값 (민감도 분석 대상)
- "확인 필요"  : 원문에서 직접 확인하지 못한 항목
"""

# ---------------------------------------------------------------------------
# 1. 고정층: Spamley 데이터셋 공개 통계
#    출처: Formisano et al. (2026), arXiv:2605.21246, Table 1
#    rho = 해당 변수와 피싱 탐지율(recall)의 Spearman 상관 (양수 = 잘 탐지)
# ---------------------------------------------------------------------------
SPAMLEY_N = 1086
SPAMLEY_MEAN_RECALL = 0.716          # 출처값 (3.1절)

# kind: cont(연속), likert(0~5 정수), binary(0/1, mean = 비율)
SPAMLEY_VARS = {
    # 이름                 kind      mean   sd     min  max   rho
    "age":               ("cont",   31.00, 13.17, 18,  75,   0.251),
    "job_experience":    ("cont",    7.78, 10.65,  0,  50,   0.241),
    "cs_knowledge":      ("likert",  3.10,  1.32,  0,   5,   0.185),
    "education":         ("likert",  2.10,  1.07,  1,   4,   0.163),
    "self_confidence":   ("likert",  3.49,  1.21,  0,   5,   0.107),
    "authority_suscept": ("likert",  3.26,  1.11,  0,   5,  -0.084),
    "agreeableness":     ("likert",  3.62,  1.12,  0,   5,  -0.079),
    "prior_attack":      ("binary",  0.16,  0.36,  0,   1,  -0.072),
    "consistency_susc":  ("likert",  2.91,  1.28,  0,   5,   0.069),
    "anti_phish_course": ("binary",  0.07,  0.26,  0,   1,   0.066),
    "emotional_stab":    ("likert",  3.00,  1.24,  0,   5,   0.061),
    "impulsivity":       ("likert",  2.12,  1.25,  0,   5,  -0.044),  # n.s. (p=0.151), 트윈 설명용으로 포함
}
# 제외한 변수와 이유
#  - Work Hours Prior to Test (rho 0.161): 피로 가설과 반대 방향. 나이·경력과의 교란 가능성이 커서
#    고정층에서 제외하고 상태층(업무 부하)과 분리.
#  - Gender (rho 0.132): 코딩 방향과 비율이 원문 표에 없음(확인 필요). 성별로 취약도를 매기는 것은
#    윤리 설계와도 맞지 않아 제외.
#  - 나머지 n.s. 변수(openness, risk perception 등): 효과가 유의하지 않아 제외.
#  - min/max 중 '_'로 표기된 연속형(age, job_experience)의 범위는 설계 가정.

# 변수 간 상관: 원문에 공개되지 않음 → 설계 가정
#  나이-경력은 강하게 얽혀 있다고 보고 0.75로 둔다. 나머지는 0 (독립 가정).
SPAMLEY_INTERCORR = {("age", "job_experience"): 0.75}

# 검증 목표 (출처값, Formisano et al. 3.4절)
CLUSTER_HIGH_RISK = {"share": 0.714, "recall": 0.688}
CLUSTER_AWARE = {"share": 0.286, "recall": 0.783}

# Spamley 시험 구조 (출처값, 2.1/2.2절): 1인당 메일 10통, 피싱 3통 미만 수신자는 제외
EMAILS_PER_USER = 10
MIN_PHISHING = 3


# ---------------------------------------------------------------------------
# 2. 상태층: 효과 크기 범위 (OR = 속을 오즈의 배수)
# ---------------------------------------------------------------------------
STATE_EFFECTS = {
    "time_pressure": {
        "or_low": 1.6, "or_mode": 2.0, "or_high": 2.8, "dist": "triangular",
        "unit": "높음 vs 낮음 (0~1 연속값으로 보간)",
        "strength": "강함",
        "basis": "환산값: Auton & Sturman (2025) F(1,195)=15.22 -> Cohen's d≈0.56 "
                 "-> Chinn(2000) lnOR = d*1.81 -> OR≈2.8 (상한). "
                 "d' 15초 1.07 vs 7초 0.45. 하한·최빈값은 설계 가정.",
        "counter": "Black & Sarno (2023): 효과 없음(바닥 효과 가능성) -> 민감도 분석에서 OR=1 시나리오로 별도 실행",
        "todo": "Butavicius et al. (2022, N=472) 효과 크기 수치 확인 필요",
    },
    "stress": {
        "or_low": 1.0, "or_high": 1.3, "dist": "uniform",
        "unit": "자가 체크 1점당",
        "strength": "중간",
        "basis": "출처값: Fallon et al. (2023), J. Information Warfare 22(2):67-85. "
                 "직원 153명 현장 연구, 업무 스트레스 1점당 반응 가능성 15% 증가 (보도자료 기준).",
        "todo": "원문 척도 범위(몇 점 만점)와 OR 신뢰구간 확인 필요",
    },
    "workload": {
        "main_or": 1.0,
        "context_weight_low": 1.0, "context_weight_high": 1.5, "dist": "uniform",
        "unit": "단독 효과 없음, 맥락 일치 점수에 곱하는 가중치",
        "strength": "탐색적",
        "basis": "Zhuo et al. (2024, N=26): 주효과 없음, 고부하에서 업무 관련 피싱 클릭 증가(사후 분석). "
                 "가중치 범위는 설계 가정.",
    },
    "sleep_deficit": {
        "or_low": 1.0, "or_high": 1.6, "dist": "uniform",
        "unit": "수면 부족 정도 0~1 (1 = 5시간 이하)",
        "strength": "간접",
        "basis": "Frenda et al. (2014): 정보 접할 때 수면 박탈이면 오정보 수용 증가, 5시간 이하 예비 연구. "
                 "피싱 직접 연구 아님. 범위는 설계 가정.",
        "todo": "효과 크기 수치 확인 필요",
    },
}
