"""
핵심 시뮬레이션 루프: 트윈(고정층+상태층+맥락층) x 공격 시나리오를 1,000회 반복해
'평소 대비 오늘의 취약도'를 분포로 계산한다.

매 반복마다 불확실한 값을 근거 범위 안에서 새로 뽑는다.
 - 고정층 계수: 200회 합성·학습 결과 중 하나
 - 상태 효과 크기: params.STATE_EFFECTS 범위
 - 맥락 효과·업무 부하 가중치: 설계 가정 범위
같은 반복 안에서 '오늘'과 '평소(상태 0, 맥락 없음)'를 함께 계산해 비율을 구한다.
"""
import json
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit

import params as P
import scenarios as S
from fixed_layer import VARS

LEVELS = [(1.25, "평소"), (1.6, "주의"), (2.2, "경계"), (float("inf"), "위험")]


@dataclass
class DayState:
    time_pressure: float = 0.0   # 0~1 (캘린더 일정 밀도에서 산출)
    stress: float = 0.0          # 평소보다 몇 점 높은가 (자가 체크 5점 척도, 0~4)
    sleep_deficit: float = 0.0   # 0~1 (1 = 5시간 이하)
    workload: float = 0.0        # 0~1
    contexts: list = field(default_factory=list)


def load_model(path="fixed_layer_model.json"):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    for k in ("mu", "sd", "coef_mean", "coef_draws", "b0_draws"):
        m[k] = np.array(m[k])
    return m


def _sample_effects(n, rng, time_pressure_null=False):
    E = P.STATE_EFFECTS
    tp = E["time_pressure"]
    return {
        "tp": np.zeros(n) if time_pressure_null else
              np.log(rng.triangular(tp["or_low"], tp["or_mode"], tp["or_high"], n)),
        "stress": np.log(rng.uniform(E["stress"]["or_low"], E["stress"]["or_high"], n)),
        "sleep": np.log(rng.uniform(E["sleep_deficit"]["or_low"], E["sleep_deficit"]["or_high"], n)),
        "wl_w": rng.uniform(E["workload"]["context_weight_low"], E["workload"]["context_weight_high"], n),
        "ctx": np.log(rng.uniform(S.CONTEXT_OR_LOW, S.CONTEXT_OR_HIGH, n)),
    }


def simulate(profile, day: DayState, model, n=1000, seed=0, time_pressure_null=False,
             effects_override=None, personal=None, contexts_scores=None):
    rng = np.random.default_rng(seed)

    # 고정층: 계수 표본 중 하나를 매 반복마다 사용
    idx = rng.integers(0, len(model["b0_draws"]), n)
    x = np.array([profile.get(v, model["mu"][j]) for j, v in enumerate(VARS)])
    xs = (x - model["mu"]) / model["sd"]
    base = model["b0_draws"][idx] + model["coef_draws"][idx] @ xs          # (n,)

    eff = _sample_effects(n, rng, time_pressure_null)
    if effects_override:                       # 민감도 분석용: 특정 효과를 고정값으로
        for k, v in effects_override.items():
            eff[k] = np.full(n, v)

    # 개인화(베이지안 갱신 결과): 개인 편차를 사후분포에서 뽑아 상태 효과에 더함
    if personal is not None and personal["n"] > 0:
        dlt = rng.multivariate_normal(personal["mean"], personal["cov"], n)
        # 근거 범위의 하한이 OR=1이므로, 개인화해도 '상태가 나쁠수록 덜 속는다'는 방향으로는 넘어가지 않게 0에서 자름
        eff["tp"] = np.maximum(eff["tp"] + dlt[:, 0], 0)
        eff["stress"] = np.maximum(eff["stress"] + dlt[:, 1], 0)
        eff["sleep"] = np.maximum(eff["sleep"] + dlt[:, 2], 0)

    state_add = (eff["tp"] * day.time_pressure + eff["stress"] * day.stress
                 + eff["sleep"] * day.sleep_deficit)
    contrib = {"시간 압박": eff["tp"] * day.time_pressure,
               "스트레스": eff["stress"] * day.stress,
               "수면 부족": eff["sleep"] * day.sleep_deficit}

    per_s, p_today_all, p_base_all = {}, [], []
    for key, sc in S.SCENARIOS.items():
        # 맥락 일치 점수: LLM 판단 결과가 있으면 사용, 없으면 규칙표
        match = (contexts_scores or {}).get(key, S.context_match(key, day.contexts))
        wl = 1 + (eff["wl_w"] - 1) * day.workload
        ctx_add = eff["ctx"] * match * wl
        other = len(sc["stages"]) - 1
        carry = expit(base + S.NON_DECISION_STAGE_OFFSET) ** other    # 결정 단계 외 통과 확률
        p_base = expit(base) * carry
        p_today = expit(base + state_add + ctx_add) * carry
        r = p_today / p_base
        per_s[key] = {"name": sc["name"], "match": match,
                      "ratio_median": float(np.median(r)),
                      "ratio_90": [float(np.percentile(r, 5)), float(np.percentile(r, 95))],
                      "ctx_contrib": ctx_add}
        p_today_all.append(p_today); p_base_all.append(p_base)

    ratio = np.mean(p_today_all, 0) / np.mean(p_base_all, 0)   # 시나리오 동일 가중 (설계 가정)
    med, lo, hi = np.median(ratio), np.percentile(ratio, 5), np.percentile(ratio, 95)

    # 설명 문장용: 오늘 취약도를 올린 요인 (log-OR 기여의 중앙값 순)
    best_ctx = max(per_s.values(), key=lambda s: np.median(s["ctx_contrib"]))
    contrib["맥락: " + best_ctx["name"]] = best_ctx["ctx_contrib"]
    drivers = sorted(((k, float(np.median(v))) for k, v in contrib.items()), key=lambda t: -t[1])
    drivers = [(k, round(float(np.exp(v)), 2)) for k, v in drivers if v > 0.01]

    top3 = sorted(per_s.values(), key=lambda s: -s["ratio_median"])[:3]
    for s in per_s.values():
        s.pop("ctx_contrib")

    return {
        "ratio_median": float(med), "ratio_90": [float(lo), float(hi)],
        "ratio_samples": ratio,
        "level": defense_level(med, hi, day),
        "drivers": drivers,
        "top3": [{k: v for k, v in s.items()} for s in top3],
        "scenarios": per_s,
    }


def defense_level(med, hi, day):
    """취약도 중앙값으로 단계를 정하고, 잃을 것이 큰 시기에는 상한 기준으로 한 단계 보수적으로."""
    names = [name for _, name in LEVELS]
    i = next(k for k, (cut, _) in enumerate(LEVELS) if med < cut)
    reason = "중앙값 기준"
    if set(day.contexts) & S.HIGH_STAKES_CONTEXTS and i < 3 and hi >= LEVELS[i][0]:
        i += 1
        reason = "잃을 것이 큰 시기 + 예측 상한이 다음 단계 → 한 단계 상향"
    return {"level": names[i], "reason": reason}


# ---------------------------------------------------------------- 시연 프리셋
DEMO_PROFILE = {   # 가상 사용자: 20대 직장 초년생 (실제 인물 아님)
    "age": 26, "job_experience": 1, "cs_knowledge": 4, "education": 3,
    "self_confidence": 4, "authority_suscept": 4, "agreeableness": 4,
    "prior_attack": 0, "consistency_susc": 3, "anti_phish_course": 0,
    "emotional_stab": 3, "impulsivity": 3,
}

PRESETS = {
    "평소의 날": DayState(),
    "택배 기다리며 마감 몰린 날": DayState(time_pressure=0.8, stress=2, sleep_deficit=0.3,
                                    workload=0.8, contexts=["parcel"]),
    "대출 준비 중 잠 못 잔 날": DayState(time_pressure=0.3, stress=1, sleep_deficit=1.0,
                                  workload=0.2, contexts=["loan"]),
}


def _print(name, res):
    lo, hi = res["ratio_90"]
    print(f"\n■ {name}")
    print(f"  평소 대비 {res['ratio_median']:.2f}배 ({lo:.2f}~{hi:.2f}배) → [{res['level']['level']}] "
          f"({res['level']['reason']})")
    if res["drivers"]:
        print("  올린 요인: " + ", ".join(f"{k} x{v}" for k, v in res["drivers"]))
    print("  주의할 공격: " + " / ".join(
        f"{s['name']} {s['ratio_median']:.2f}배" for s in res["top3"]))


if __name__ == "__main__":
    model = load_model()
    for name, day in PRESETS.items():
        _print(name, simulate(DEMO_PROFILE, day, model))
    print("\n[민감도] 시간 압박 효과 없음(Black & Sarno 2023) 시나리오")
    _print("택배 기다리며 마감 몰린 날 (시간 압박 OR=1)",
           simulate(DEMO_PROFILE, PRESETS["택배 기다리며 마감 몰린 날"], model, time_pressure_null=True))
