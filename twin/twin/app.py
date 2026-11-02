"""
상태 적응형 사회공학 디지털 트윈 — 본선 시연 대시보드
실행: streamlit run app.py

시제품은 가상의 사용자·가상의 일정·가상의 반응 기록만 사용한다.
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit

import params as P
import scenarios as S
import sensitivity
from bayes_update import update, synthetic_log, describe
from context_llm import llm_scores, rule_scores
from fixed_layer import VARS, base_logit
from simulate import DayState, DEMO_PROFILE, load_model, simulate

st.set_page_config(page_title="오늘의 피싱 예보", page_icon="🛡️", layout="wide")

# Streamlit Cloud: Settings → Secrets 에 ANTHROPIC_API_KEY 를 넣으면 LLM 맥락 판단이 켜진다
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:                               # secrets 파일이 없는 로컬 실행
    pass

LEVEL_COLOR = {"평소": "#2e9d5b", "주의": "#e0a21a", "경계": "#e46a2b", "위험": "#d23b3b"}
LEVEL_ACTION = {
    "평소": "의심 문자·전화에 일반 표시",
    "주의": "오늘 주의할 공격 유형과 관련된 문자·전화에 경고 강화",
    "경계": "모르는 번호 통화 중 실시간 경고, 링크 클릭 전 확인",
    "위험": "송금 앱 실행 시 지연 권고, 미리 동의한 보호자에게 알림",
}
SURVEY = {  # 고정층 설문 (화면 표시 이름, 최소, 최대, 단계)
    "age": ("나이", 18, 75, 1), "job_experience": ("경력(년)", 0, 40, 1),
    "cs_knowledge": ("컴퓨터 지식 (0~5)", 0, 5, 1), "education": ("학력 (1~4)", 1, 4, 1),
    "self_confidence": ("피싱 구별 자신감 (0~5)", 0, 5, 1),
    "authority_suscept": ("권위에 따르는 편 (0~5)", 0, 5, 1),
    "agreeableness": ("친화성 (0~5)", 0, 5, 1), "prior_attack": ("피싱 피해 경험 (0/1)", 0, 1, 1),
    "consistency_susc": ("일관성 성향 (0~5)", 0, 5, 1), "anti_phish_course": ("피싱 교육 수강 (0/1)", 0, 1, 1),
    "emotional_stab": ("정서 안정성 (0~5)", 0, 5, 1), "impulsivity": ("충동성 (0~5)", 0, 5, 1),
}
PRESETS = {   # 시연 순서: 평소 → 주의 → 경계 → 위험 (simulate.PRESETS와 같은 값)
    "평소의 날": dict(tp=0.0, stress=0, sleep=0.0, wl=0.0, events=""),
    "마감이 몰린 날": dict(tp=0.5, stress=1, sleep=0.3, wl=0.3, events=""),
    "택배 기다리는 날": dict(tp=0.3, stress=0, sleep=0.0, wl=0.3,
                       events="다음 주 화요일 쿠팡 택배 도착 예정"),
    "택배 기다리며 마감 몰린 날": dict(tp=0.8, stress=2, sleep=0.3, wl=0.8,
                                 events="다음 주 화요일 쿠팡 택배 도착 예정"),
    "대출 준비 중 잠 못 잔 날": dict(tp=0.3, stress=1, sleep=1.0, wl=0.2,
                                events="전세 계약 앞두고 대출 알아보는 중\n주말 이사"),
}
DEFAULT_PRESET = "택배 기다리며 마감 몰린 날"   # 첫 화면: 시뮬레이션이 움직이는 모습이 바로 보이도록


# ---------------------------------------------------------------- 캐시된 계산
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def run_sim(profile_t, tp, stress, sleep, wl, ctx_keys, scores_t, tp_null, personal_days):
    model = get_model()
    profile = dict(profile_t)
    day = DayState(tp, stress, sleep, wl, list(ctx_keys))
    personal = None
    if personal_days:
        b = base_logit(profile, model)
        personal = update(synthetic_log(personal_days, np.array([0, 0, -0.8]), b,
                                        np.random.default_rng(7)), b)
    res = simulate(profile, day, model, contexts_scores=dict(scores_t),
                   time_pressure_null=tp_null, personal=personal)
    res["ratio_samples"] = res["ratio_samples"].tolist()
    return res


@st.cache_data(show_spinner=False)
def get_scores(events_t):
    return llm_scores(list(events_t))


@st.cache_data(show_spinner=False)
def run_sensitivity(profile_t, tp, stress, sleep, wl, ctx_keys, scores_t):
    day = DayState(tp, stress, sleep, wl, list(ctx_keys))
    return sensitivity.run(dict(profile_t), day, get_model(), contexts_scores=dict(scores_t))


@st.cache_data(show_spinner=False)
def personalization_curve(profile_t, tp, stress, sleep, wl, ctx_keys, scores_t):
    model = get_model()
    profile = dict(profile_t)
    b = base_logit(profile, model)
    day = DayState(tp, stress, sleep, wl, list(ctx_keys))
    rows, last = [], None
    for d in (0, 30, 60, 90, 120, 150, 180):
        post = update(synthetic_log(d, np.array([0, 0, -0.8]), b, np.random.default_rng(7)), b)
        r = simulate(profile, day, model, contexts_scores=dict(scores_t), personal=post)
        rows.append({"사용 일수": d, "중앙값": r["ratio_median"],
                     "하한": r["ratio_90"][0], "상한": r["ratio_90"][1]})
        last = post
    return pd.DataFrame(rows), describe(update([], b)), describe(last)


# ---------------------------------------------------------------- 사이드바 입력
def apply_preset(name):
    p = PRESETS[name]
    st.session_state.update(tp=p["tp"], stress=p["stress"], sleep=p["sleep"], wl=p["wl"],
                            events=p["events"])


for k, v in PRESETS[DEFAULT_PRESET].items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.markdown("### 시연 프리셋")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), width="stretch")

    st.markdown("### 오늘의 상태")
    tp = st.slider("시간 압박 (캘린더 일정 밀도)", 0.0, 1.0, key="tp", step=0.1)
    stress = st.slider("스트레스 (평소보다 몇 점 높은가)", 0, 4, key="stress")
    sleep = st.slider("수면 부족 (1 = 5시간 이하)", 0.0, 1.0, key="sleep", step=0.1)
    wl = st.slider("업무량", 0.0, 1.0, key="wl", step=0.1)

    st.markdown("### 요즘 일정 (맥락)")
    events = st.text_area("한 줄에 하나씩", key="events", height=90)

    with st.expander("고정층 설문 (가상 사용자)"):
        profile = {}
        for v, (label, lo, hi, step) in SURVEY.items():
            profile[v] = st.slider(label, lo, hi, int(DEMO_PROFILE[v]), step, key=f"pf_{v}")

    st.markdown("### 옵션")
    tp_null = st.toggle("시간 압박 효과 없음 가정 (Black & Sarno 2023)")
    personal_days = st.select_slider("개인화: 반응 기록 사용 일수", [0, 30, 90, 180], value=0)

# ---------------------------------------------------------------- 계산
event_list = tuple(e for e in events.splitlines() if e.strip())
scores, reasons, score_src = get_scores(event_list)
_, ctx_keys = rule_scores(list(event_list))
profile_t = tuple(sorted(profile.items()))
scores_t = tuple(sorted(scores.items()))
args = (profile_t, tp, stress, sleep, wl, tuple(ctx_keys), scores_t)

res = run_sim(*args, tp_null, personal_days)
level = res["level"]["level"]
med, (lo, hi) = res["ratio_median"], res["ratio_90"]

model = get_model()
mean_profile = {v: model["mu"][j] for j, v in enumerate(VARS)}
base_or = float(np.exp(base_logit(profile, model) - base_logit(mean_profile, model)))
base_p = float(expit(base_logit(profile, model)))


def josa(word, with_batchim, without):
    """마지막 글자 받침에 따라 조사 선택 (예: 스미싱'이 / 안내'가)."""
    ch = word.strip()[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    return without


# (입력값 기준, 연결형, 이유형) — 실제로 '몰렸다·부족하다'고 말할 수 있을 때만 문장에 넣는다
STATE_PHRASES = {
    "시간 압박": (lambda: tp >= 0.5, "일정이 몰려 있고", "일정이 몰려 있어서"),
    "스트레스": (lambda: stress >= 1, "스트레스가 평소보다 높고", "스트레스가 평소보다 높아서"),
    "수면 부족": (lambda: sleep >= 0.5, "잠이 부족하고", "잠이 부족해서"),
}


def explain(drivers, med):
    if not drivers or med < 1.1:
        return "평소와 비슷한 날이에요. 기본 표시만 유지할게요."
    ctx = [k[4:] for k, _ in drivers if k.startswith("맥락: ")]
    states = [k for k, _ in drivers if k in STATE_PHRASES and STATE_PHRASES[k][0]()][:2]
    out = []
    if ctx:
        out.append(f"요즘 일정 때문에 '{ctx[0]}'{josa(ctx[0], '이', '가')} 진짜처럼 느껴지기 쉬워요.")
    if states:
        parts = [STATE_PHRASES[k][1] for k in states[:-1]] + [STATE_PHRASES[states[-1]][2]]
        out.append(f"오늘은 {' '.join(parts)} 서두르기 쉬운 날이에요.")
    return " ".join(out) + " 링크와 송금 요청은 한 번 더 확인하세요."


# ---------------------------------------------------------------- 화면
st.markdown("## 🛡️ 오늘의 피싱 예보")
st.caption("상태 적응형 사회공학 디지털 트윈 · 시제품은 가상 사용자와 가상 기록만 사용합니다")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["오늘의 예보", "시뮬레이션", "방어 단계", "개인화(동기화)", "근거·민감도"])

with tab1:
    c1, c2 = st.columns([1, 1.2], gap="large")
    color = LEVEL_COLOR[level]
    pos = min(max((med - 1) / 2, 0), 1) * 100
    with c1:
        top3 = [s for s in res["top3"] if s["ratio_median"] >= 1.1]
        top_html = "".join(
            f"<div style='padding:8px 0;border-top:1px solid #eee'><b>{i+1}. {s['name']}</b>"
            f"<span style='float:right;color:#666'>{s['ratio_median']:.1f}배</span></div>"
            for i, s in enumerate(top3)) or (
            "<div style='padding:8px 0;border-top:1px solid #eee;color:#666'>"
            "오늘은 특별히 주의할 공격이 없어요</div>")
        st.markdown(f"""
<div style="max-width:380px;border:10px solid #1f2633;border-radius:36px;padding:22px 18px;background:#f5f6f8">
  <div style="color:#888;font-size:13px">오늘의 피싱 예보</div>
  <div style="background:white;border-radius:16px;padding:16px;margin-top:10px">
    <div style="display:flex;align-items:center;gap:14px">
      <div style="background:{color};color:white;font-weight:700;border-radius:10px;padding:10px 14px;font-size:20px">{level}</div>
      <div><div style="color:#666;font-size:13px">평소 대비 위험도</div>
      <div style="font-size:26px;font-weight:800">약 {med:.1f}배 <span style="font-size:13px;color:#888">({lo:.1f}~{hi:.1f}배)</span></div></div>
    </div>
    <div style="position:relative;height:10px;border-radius:5px;margin:16px 0 4px;
      background:linear-gradient(90deg,#2e9d5b,#e0a21a,#e46a2b,#d23b3b)">
      <div style="position:absolute;left:{pos}%;top:-5px;width:4px;height:20px;background:#1f2633;border-radius:2px"></div>
    </div>
    <div style="display:flex;justify-content:space-between;color:#999;font-size:11px"><span>평소</span><span>2배</span><span>3배+</span></div>
    <div style="margin-top:12px;font-size:14px;line-height:1.5">{explain(res['drivers'], med)}</div>
  </div>
  <div style="background:white;border-radius:16px;padding:12px 16px;margin-top:10px">
    <div style="font-weight:700;margin-bottom:4px">오늘 특히 주의할 공격</div>{top_html}
  </div>
  <div style="text-align:center;color:#2e9d5b;font-size:12px;margin-top:10px">🔒 모든 분석은 이 기기 안에서만 이뤄져요</div>
</div>""", unsafe_allow_html=True)
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("오늘 취약도", f"{med:.2f}배", help=f"90% 범위 {lo:.2f}~{hi:.2f}배")
        m2.metric("방어 단계", level)
        m3.metric("기본 취약도(고정층)", f"{base_or:.2f}배", help="같은 조건의 평균 사용자 대비 속을 오즈 (Spamley 기반)")
        st.caption(f"오늘 취약도 90% 범위: {lo:.2f}~{hi:.2f}배 · 기본 취약도는 평균 사용자 대비 · "
                   f"이 사람의 평소 속을 확률(추정) {base_p:.0%}")
        st.caption("오늘 취약도는 '이 사람의 평소' 대비 값입니다. 평소 잘 속지 않는 사람일수록 같은 상태 변화에도 "
                   "배수가 크게 나오고, 원래 취약한 사람은 배수는 작아도 절대 수준이 높습니다. "
                   "그래서 두 값을 함께 봅니다.")
        st.markdown("**오늘 수치를 올린 요인** (속을 오즈 배수, 중앙값)")
        if res["drivers"]:
            st.dataframe(pd.DataFrame(res["drivers"], columns=["요인", "배수"]), hide_index=True,
                         width="stretch")
            st.caption("요인 배수는 '속을 오즈' 기준이라 곱한 값이 오늘 취약도와 같지 않습니다. "
                       "오늘 취약도는 확률 기준으로 바꾼 뒤 공격 4종을 평균한 값이고, "
                       "맥락 요인은 맞물리는 공격에만 적용됩니다.")
        else:
            st.write("없음")
        st.markdown(f"**맥락 판단** ({'LLM' if score_src == 'llm' else '키워드 규칙 · 기기 내 경량 판단'})")
        st.dataframe(pd.DataFrame([{"공격 유형": S.SCENARIOS[k]["name"], "맞물림": round(v, 2),
                                    "이유": reasons.get(k, "")} for k, v in scores.items()]),
                     hide_index=True, width="stretch")
        if score_src != "llm" and event_list:
            st.caption("LLM 연결이 없을 때는 키워드 규칙으로 자동 대체합니다 (목표 서비스는 기기 내 소형 모델).")
        st.caption(f"단계 결정: {res['level']['reason']} · 절대 확률이 아닌 '같은 사람의 평소 대비' 값입니다.")

with tab2:
    st.markdown(f"**1,000회 반복 시뮬레이션 분포** — 매 회 효과 크기를 근거 범위에서 새로 뽑습니다")
    df = pd.DataFrame({"평소 대비 배수": res["ratio_samples"]})
    hist = alt.Chart(df).mark_bar(color="#5b7bd5", opacity=0.85).encode(
        alt.X("평소 대비 배수:Q", bin=alt.Bin(maxbins=40)), alt.Y("count()", title="반복 수"))
    rule = alt.Chart(pd.DataFrame({"x": [med]})).mark_rule(color=color, size=3).encode(x="x:Q")
    st.altair_chart(hist + rule, width="stretch")
    st.caption(f"중앙값 {med:.2f}배 · 90% 범위 {lo:.2f}~{hi:.2f}배. 범위가 넓으면 근거가 약한 요인이 크게 작용한 날입니다.")

    sc = pd.DataFrame([{"공격": s["name"], "중앙값": s["ratio_median"], "하한": s["ratio_90"][0],
                        "상한": s["ratio_90"][1]} for s in res["scenarios"].values()])
    base = alt.Chart(sc).encode(alt.Y("공격:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)))
    st.markdown("**공격 유형별 평소 대비 배수** (선: 90% 범위)")
    st.altair_chart(base.mark_rule(size=2, color="#999").encode(x=alt.X("하한:Q", title="평소 대비 배수", scale=alt.Scale(zero=False)),
                                                                 x2="상한:Q")
                    + base.mark_point(filled=True, size=120, color="#1f2633").encode(x="중앙값:Q"),
                    width="stretch")

with tab3:
    st.markdown("**같은 의심 문자, 다른 대응** — 기존 서비스는 모든 날 같은 기준, 본 서비스는 그날의 취약도에 따라 강도를 바꿉니다")
    cols = st.columns(4)
    for col, name in zip(cols, ["평소", "주의", "경계", "위험"]):
        on = name == level
        border = f"3px solid {LEVEL_COLOR[name]}" if on else "1px solid #ddd"
        extra = {
            "평소": "<span style='background:#fff3cd;border-radius:6px;padding:2px 6px;font-size:12px'>⚠ 의심 문자</span>",
            "주의": "<div style='background:#fff3cd;border-radius:8px;padding:8px;font-size:12px'>오늘 주의할 공격과 같은 유형이에요</div>",
            "경계": "<div style='background:#fde2d3;border-radius:8px;padding:8px;font-size:12px'><b>잠깐, 오늘은 이 문자에 취약한 날이에요</b><br>✓ 공식 앱에서 직접 조회<br>✓ 문자 속 링크는 열지 않기<br><i>링크 열기는 5초 후 활성화</i></div>",
            "위험": "<div style='background:#f8d0d0;border-radius:8px;padding:8px;font-size:12px'><b>송금 앱 실행 시 10분 지연 권고</b><br>미리 동의한 보호자에게 알림<br>✓ 공식 앱에서 직접 조회</div>",
        }[name]
        col.markdown(f"""
<div style="border:{border};border-radius:18px;padding:12px;min-height:290px;background:{'#fff' if on else '#fafafa'}">
  <div style="font-weight:700;color:{LEVEL_COLOR[name]}">{name}{' ← 오늘' if on else ''}</div>
  <div style="font-size:12px;color:#666;margin:4px 0 8px">{LEVEL_ACTION[name]}</div>
  <div style="background:#eef0f3;border-radius:10px;padding:8px;font-size:12px;margin-bottom:8px">
    [국제발신] 고객님 택배가 주소 불일치로 보류되었습니다. 확인: ●●●●●.●●</div>
  {extra}
</div>""", unsafe_allow_html=True)
    st.caption("※ 목표 서비스 화면 예시. 문자 내용과 주소는 오남용 방지를 위해 가려서 표시합니다.")
    st.markdown("**단계 결정 규칙**: 중앙값 1.25배 미만 평소 · 1.6배 미만 주의 · 2.2배 미만 경계 · 그 이상 위험. "
                "대출 준비·이사처럼 잃을 것이 큰 시기에 예측 상한이 다음 단계를 넘으면 한 단계 올립니다.")

with tab4:
    st.markdown("**현실과의 동기화** — 실제 반응(신고·무시·열어봄)으로 이 사람에게 맞게 값을 고칩니다")
    st.caption("시연 시나리오: 잠이 부족한 날에도 의심 문자를 꾸준히 신고하는 가상 사용자 (하루 평균 1건의 가상 기록)")
    curve, before, after = personalization_curve(*args)
    band = alt.Chart(curve).mark_area(opacity=0.2, color="#5b7bd5").encode(
        x=alt.X("사용 일수:Q"), y=alt.Y("하한:Q", title="오늘 취약도 (평소 대비)", scale=alt.Scale(zero=False)), y2="상한:Q")
    line = alt.Chart(curve).mark_line(point=True, color="#1f2633").encode(x="사용 일수:Q", y="중앙값:Q")
    st.altair_chart(band + line, width="stretch")
    cA, cB = st.columns(2)
    cA.markdown("**처음 (선행 연구 평균값)**")
    cA.dataframe(pd.DataFrame(before).rename(columns={"factor": "요인", "population_or": "평균 효과",
                 "personal_or": "이 사람", "certainty": "확실도"}), hide_index=True, width="stretch")
    cB.markdown("**180일 후**")
    cB.dataframe(pd.DataFrame(after).rename(columns={"factor": "요인", "population_or": "평균 효과",
                 "personal_or": "이 사람", "certainty": "확실도"}), hide_index=True, width="stretch")
    st.caption("한계: 의심 메시지가 드물면 개인화에 수개월이 걸립니다. 반응 기록은 기기 밖으로 나가지 않는다는 전제입니다.")

with tab5:
    center, rows = run_sensitivity(*args)
    st.markdown(f"**민감도 분석 — 어떤 가정이 결과를 가장 크게 바꾸나** (모든 가정 중앙값일 때 {center:.2f}배)")
    sd = pd.DataFrame(rows)
    sd["lo"] = sd[["하한", "상한"]].min(axis=1)
    sd["hi"] = sd[["하한", "상한"]].max(axis=1)
    sd["라벨"] = sd["가정"] + " [" + sd["근거"] + "]"
    tor = alt.Chart(sd).mark_bar(color="#e46a2b", height=18).encode(
        y=alt.Y("라벨:N", sort=list(sd["라벨"]), title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("lo:Q", title="오늘 취약도 (평소 대비)", scale=alt.Scale(zero=False)), x2="hi:Q")
    cl = alt.Chart(pd.DataFrame({"x": [center]})).mark_rule(color="#1f2633", strokeDash=[4, 3]).encode(x="x:Q")
    st.altair_chart(tor + cl, width="stretch")
    st.caption("막대가 길수록 결과가 그 가정에 기대고 있습니다. 근거가 약한 가정이 위에 있다면 우선 검증 대상입니다.")
    st.caption("앞 단계 통과 가정: 보이스피싱은 접근→신뢰 형성→압박→송금 요구 단계를 거칩니다. 상태 효과는 근거 연구처럼 "
               "판단이 갈리는 '압박' 단계에만 적용하고, 나머지 단계는 '한 번 응한 사람은 다음 단계로 넘어가기 쉽다'고 가정합니다.")

    st.markdown("**상태층 근거표**")
    st.dataframe(pd.DataFrame([{
        "요인": {"time_pressure": "시간 압박", "stress": "스트레스", "workload": "업무 부하", "sleep_deficit": "수면 부족"}[k], "범위(OR)": f"{v.get('or_low', v.get('context_weight_low'))} ~ {v.get('or_high', v.get('context_weight_high'))}",
        "단위": v["unit"], "근거 강도": v["strength"], "근거": v["basis"], "확인 필요": v.get("todo", "")}
        for k, v in P.STATE_EFFECTS.items()]), hide_index=True, width="stretch")
    st.markdown(f"**맥락 일치 효과**: OR {S.CONTEXT_OR_LOW}~{S.CONTEXT_OR_HIGH} (설계 가정, 수치 근거는 확인한 범위에서 없음)")
    st.markdown(f"**고정층**: Spamley 공개 통계(N={P.SPAMLEY_N}, 평균 탐지율 {P.SPAMLEY_MEAN_RECALL}) 기반 합성 데이터로 학습 · "
                "변수 12개의 탐지율 상관을 오차 0.01 이내로 재현")
