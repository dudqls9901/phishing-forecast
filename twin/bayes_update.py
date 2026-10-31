"""
현실과의 동기화: 사용자의 실제 반응 기록으로 '이 사람에게서 각 상태 요인이 얼마나 작용하는지'를 갱신한다.

모델
  logit P(열어봄) = 고정층 기본값 + Σ_f (모집단 효과_f + 개인 편차_f) * 상태_f + 맥락 항
  개인 편차_f ~ N(0, PRIOR_SD^2)  ← 처음에는 0 (= 선행 연구 평균값 그대로)

반응 기록이 쌓이면 라플라스 근사(뉴턴법)로 개인 편차의 사후분포(평균, 공분산)를 구하고,
시뮬레이션은 매 반복마다 이 사후분포에서 개인 편차를 뽑아 상태 효과에 더한다.

반응 정의: 신고·무시 = 0 (속지 않음), 열어봄(링크 클릭·회신) = 1
기록은 기기 밖으로 나가지 않는다는 전제. 시제품에서는 가상의 반응 기록만 사용.
"""
import numpy as np
from scipy.special import expit

import params as P
import scenarios as S

FACTORS = ["time_pressure", "stress", "sleep_deficit"]
FACTOR_KO = {"time_pressure": "시간 압박", "stress": "스트레스", "sleep_deficit": "수면 부족"}
PRIOR_SD = 0.5     # 설계 가정: 개인 차이가 모집단 효과에서 OR 기준 약 x0.37~x2.7 (2SD) 안에 있다고 봄


def population_logor():
    """모집단 효과의 중심값 (log-OR). 상태 효과 범위의 중앙."""
    E = P.STATE_EFFECTS
    tp = E["time_pressure"]
    return np.log(np.array([
        (tp["or_low"] + tp["or_mode"] + tp["or_high"]) / 3,
        (E["stress"]["or_low"] + E["stress"]["or_high"]) / 2,
        (E["sleep_deficit"]["or_low"] + E["sleep_deficit"]["or_high"]) / 2,
    ]))


def _design(records):
    X = np.array([[r[f] for f in FACTORS] for r in records], float)
    y = np.array([r["opened"] for r in records], float)
    return X, y


def update(records, base_logit, ctx_logor_mid=np.log(2.25)):
    """
    records: [{"time_pressure":..,"stress":..,"sleep_deficit":..,"scenario":key,"contexts":[..],"opened":0/1}, ...]
    base_logit: 이 사용자의 고정층 기본 logit
    반환: {"mean": δ 평균(3,), "cov": 공분산(3x3), "n": 기록 수}
    """
    if not records:
        return {"mean": np.zeros(3), "cov": np.eye(3) * PRIOR_SD ** 2, "n": 0}
    X, y = _design(records)
    beta = population_logor()
    ctx = np.array([S.context_match(r["scenario"], r.get("contexts", [])) for r in records]) * ctx_logor_mid
    offset = base_logit + X @ beta + ctx

    d = np.zeros(3)
    prior_prec = np.eye(3) / PRIOR_SD ** 2
    for _ in range(50):                       # 뉴턴법으로 사후 최빈값 탐색
        p = expit(offset + X @ d)
        grad = X.T @ (y - p) - prior_prec @ d
        H = -(X.T * (p * (1 - p))) @ X - prior_prec
        step = np.linalg.solve(H, grad)
        d -= step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(-H)
    return {"mean": d, "cov": cov, "n": len(records)}


def describe(post):
    """사람이 읽을 수 있는 요약: 이 사람에게서 각 요인의 효과가 평균 대비 몇 배인지."""
    beta = population_logor()
    out = []
    for i, f in enumerate(FACTORS):
        pop_or = np.exp(beta[i])
        personal_or = np.exp(beta[i] + post["mean"][i])
        sd = np.sqrt(post["cov"][i, i])
        out.append({"factor": FACTOR_KO[f], "population_or": round(float(pop_or), 2),
                    "personal_or": round(float(personal_or), 2),
                    "certainty": round(float(1 - sd / PRIOR_SD), 2)})   # 0 = 사전과 같음, 1에 가까울수록 확실
    return out


# ---------------------------------------------------------------- 시연용 가상 반응 기록
def synthetic_log(days, true_delta, base_logit, rng, msgs_per_day=1.0):
    """'진짜 개인 편차'를 정해두고 가상의 반응 기록을 만든다 (갱신이 제대로 수렴하는지 확인용)."""
    beta = population_logor()
    recs = []
    keys = list(S.SCENARIOS)
    for _ in range(days):
        st = {"time_pressure": rng.uniform(0, 1), "stress": rng.integers(0, 4),
              "sleep_deficit": rng.choice([0, 0, 0.5, 1.0])}
        for _ in range(rng.poisson(msgs_per_day)):
            x = np.array([st[f] for f in FACTORS])
            p = expit(base_logit + x @ (beta + true_delta))
            recs.append({**st, "scenario": rng.choice(keys), "contexts": [],
                         "opened": int(rng.random() < p)})
    return recs


if __name__ == "__main__":
    from simulate import load_model, DEMO_PROFILE
    from fixed_layer import base_logit as fixed_base
    rng = np.random.default_rng(7)
    model = load_model()
    b = fixed_base(DEMO_PROFILE, model)

    # 기획서 예시: 잠이 부족한 날에도 꾸준히 신고하는 사용자 → 수면 효과가 작아져야 함
    true_delta = np.array([0.0, 0.0, -0.8])
    print("기획서 예시: 수면 부족에 강한 사용자 (진짜 개인 편차 = 수면 log-OR -0.8)")
    for days in (0, 30, 90, 180):
        post = update(synthetic_log(days, true_delta, b, rng), b)
        s = "  |  ".join(f"{d['factor']} x{d['personal_or']} (확실도 {d['certainty']})"
                         for d in describe(post))
        print(f"  {days:>3}일, 기록 {post['n']:>3}건: {s}")
