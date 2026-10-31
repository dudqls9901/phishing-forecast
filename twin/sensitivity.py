"""
민감도 분석: '어떤 가정이 결과를 가장 크게 바꾸는가'

각 불확실한 값을 범위의 하한·상한으로 바꿔 보고(나머지는 범위 중앙에 고정),
오늘의 취약도(평소 대비 배수)가 얼마나 움직이는지 비교한다. 움직임이 클수록 결과가 그 가정에 기대고 있다.
"""
import numpy as np

import params as P
import scenarios as S
from simulate import simulate

E = P.STATE_EFFECTS

# (표시 이름, 근거 강도, override 키, 하한, 상한)
ITEMS = [
    ("시간 압박 효과", "강함 (반대 연구 1건)", "tp", np.log(1.0), np.log(E["time_pressure"]["or_high"])),
    ("스트레스 효과", "중간", "stress", np.log(E["stress"]["or_low"]), np.log(E["stress"]["or_high"])),
    ("수면 부족 효과", "간접 (설계 가정)", "sleep", np.log(E["sleep_deficit"]["or_low"]), np.log(E["sleep_deficit"]["or_high"])),
    ("업무 부하 가중치", "탐색적 (설계 가정)", "wl_w", E["workload"]["context_weight_low"], E["workload"]["context_weight_high"]),
    ("맥락 일치 효과", "설계 가정", "ctx", np.log(S.CONTEXT_OR_LOW), np.log(S.CONTEXT_OR_HIGH)),
]
STAGE_ITEM = ("비결정 단계 통과 보정", "설계 가정", 0.0, 2.0)


def _mid():
    return {
        "tp": np.log(E["time_pressure"]["or_mode"]),
        "stress": np.log((E["stress"]["or_low"] + E["stress"]["or_high"]) / 2),
        "sleep": np.log((E["sleep_deficit"]["or_low"] + E["sleep_deficit"]["or_high"]) / 2),
        "wl_w": (E["workload"]["context_weight_low"] + E["workload"]["context_weight_high"]) / 2,
        "ctx": np.log((S.CONTEXT_OR_LOW + S.CONTEXT_OR_HIGH) / 2),
    }


def run(profile, day, model, n=400, contexts_scores=None):
    mid = _mid()
    kw = dict(n=n, seed=1, contexts_scores=contexts_scores)
    center = simulate(profile, day, model, effects_override=mid, **kw)["ratio_median"]
    rows = []
    for name, strength, key, lo, hi in ITEMS:
        r_lo = simulate(profile, day, model, effects_override={**mid, key: lo}, **kw)["ratio_median"]
        r_hi = simulate(profile, day, model, effects_override={**mid, key: hi}, **kw)["ratio_median"]
        rows.append({"가정": name, "근거": strength, "하한": r_lo, "상한": r_hi, "폭": abs(r_hi - r_lo)})

    name, strength, lo, hi = STAGE_ITEM
    orig = S.NON_DECISION_STAGE_OFFSET
    out = []
    for v in (lo, hi):
        S.NON_DECISION_STAGE_OFFSET = v
        out.append(simulate(profile, day, model, effects_override=mid, **kw)["ratio_median"])
    S.NON_DECISION_STAGE_OFFSET = orig
    rows.append({"가정": name, "근거": strength, "하한": out[0], "상한": out[1], "폭": abs(out[1] - out[0])})

    rows.sort(key=lambda r: -r["폭"])
    return center, rows


if __name__ == "__main__":
    from simulate import load_model, DEMO_PROFILE, PRESETS
    m = load_model()
    for pname in ("택배 기다리며 마감 몰린 날", "대출 준비 중 잠 못 잔 날"):
        c, rows = run(DEMO_PROFILE, PRESETS[pname], m)
        print(f"\n■ {pname} (모든 가정 중앙값일 때 {c:.2f}배)")
        for r in rows:
            print(f"  {r['가정']:<14} [{r['근거']}]  {r['하한']:.2f} ~ {r['상한']:.2f}배  (폭 {r['폭']:.2f})")
