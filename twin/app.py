"""
상태 적응형 사회공학 디지털 트윈 — 본선 시연 화면
실행: streamlit run app.py

앞면: 오늘의 상태(프리셋 3개) · 스마트폰 · 도착한 메시지(3장 + 직접 입력)
뒷면: 폰을 누르면 넓은 근거 화면 (시뮬레이션 · 근거·민감도 · 개인화 · AI 비서 연동 · 직접 조정)
이전 대시보드는 app_legacy.py 에 그대로 남겨 두었다.

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
from context_llm import llm_scores, llm_profile, rule_scores, classify_message
from fixed_layer import VARS, base_logit
from simulate import DayState, DEMO_PROFILE, LEVELS, load_model, simulate
from stage import stage

st.set_page_config(page_title="오늘의 피싱 예보", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

# Streamlit Cloud: Settings → Secrets 에 ANTHROPIC_API_KEY 를 넣으면 LLM 판단이 켜진다
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:                               # secrets 파일이 없는 로컬 실행
    pass

# 발표장 화면용: Streamlit 기본 메뉴·툴바·여백 정리, 뒷면 등장 애니메이션
st.markdown("""
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], footer, [data-testid="stSidebarCollapsedControl"]{display:none !important}
.block-container{padding-top:1.4rem !important;padding-bottom:2rem !important;max-width:1280px}
.st-key-back{animation:backin .35s ease-out;transform-origin:center}
@keyframes backin{from{transform:perspective(1400px) rotateY(-90deg);opacity:.4}to{transform:none;opacity:1}}
.brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.brand b{font-size:26px}
.brand span{color:#6b7280;font-size:14px}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- 상수
LEVEL_STYLE = {   # 단계: (색, 배경, 날씨 이모지, 날씨, AI 비서 대응)
    "평소": ("#2e9d5b", "#e8f5ec", "☀️", "맑음", "평소처럼 '의심 문자' 표시만 할게요."),
    "주의": ("#c98a00", "#fdf4dc", "⛅", "구름 조금", "오늘 주의할 유형이에요. 경고를 크게 띄울게요."),
    "경계": ("#e46a2b", "#fde9dd", "🌧️", "비", "잠깐, 오늘은 이 유형에 약한 날이에요. 링크는 5초 뒤에 열 수 있어요."),
    "위험": ("#d23b3b", "#fbe3e3", "⛈️", "폭풍", "링크를 잠시 막아 둘게요. 송금 앱을 열면 10분 기다리도록 권할게요."),
}
LEVEL_COLOR = {k: v[0] for k, v in LEVEL_STYLE.items()}
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
PRESETS = [   # 앞면 상태 버튼 3개 (simulate.PRESETS 와 같은 값)
    dict(label="평온한 날", sub="잘 자고, 일정에 여유가 있는 날",
         tp=0.0, stress=0, sleep=0.0, wl=0.0, events=""),
    dict(label="마감 + 택배 대기", sub="일정이 몰리고, 택배를 기다리는 중",
         tp=0.8, stress=2, sleep=0.3, wl=0.8, events="다음 주 화요일 쿠팡 택배 도착 예정"),
    dict(label="대출 준비 + 잠 못 잔 날", sub="전세 대출을 알아보며 이사 준비 중",
         tp=0.3, stress=1, sleep=1.0, wl=0.2, events="전세 계약 앞두고 대출 알아보는 중\n주말 이사"),
]
CARDS = [   # 앞면 메시지 카드 3장 (가상 문자 · 주소는 가려서 표시)
    dict(key="parcel_smishing", icon="📦", title="택배 주소 확인 문자", from_="[국제발신] 문자",
         text="고객님 택배가 주소 불일치로 보류되었습니다. 주소 확인: ●●●●.●●/●●"),
    dict(key="family_impersonation", icon="💬", title="가족 사칭 메신저", from_="모르는 번호 · 메신저",
         text="엄마 나 폰 액정 깨져서 이 번호로 연락해. 급하게 결제할 게 있는데 잠깐 도와줄 수 있어?"),
    dict(key="loan_fraud", icon="🏦", title="저금리 대환대출 문자", from_="[Web발신] 문자",
         text="고객님은 정부지원 저금리 대환대출 대상입니다. 오늘 마감, 한도 조회: ●●●●.●●"),
]

# ---------------------------------------------------------------- 상태 초기화
ss = st.session_state
ADJ_KEYS = ["tp", "stress", "sleep", "wl", "events", "tp_null", "personal_days", "memo"] + [f"pf_{v}" for v in SURVEY]
for k in ADJ_KEYS:          # 뒷면에만 있는 위젯 값이 앞면에서 지워지지 않도록 유지
    if k in ss:
        ss[k] = ss[k]

SAMPLE_MEMO = """[AI 비서 메모 예시 · 가상 인물]
- 52세, 동네에서 18년째 작은 식당을 운영함
- 스마트폰은 카톡과 배달앱 정도만 쓰고, 앱 설정은 딸이 대신 해 줌
- 고등학교 졸업
- 은행이나 관공서에서 연락이 오면 일단 시키는 대로 하는 편이라고 말함
- 작년에 '주소 확인' 문자 링크를 눌렀다가 카드를 재발급한 적 있음
- 보안 교육은 받아 본 적 없음
- 한 번 하겠다고 한 일은 끝까지 하는 성격
- 성격이 급해서 확인하기 전에 먼저 움직이는 편
- 요즘 가게를 넓히려고 은행 상담을 다니는 중
- 딸이 다음 주에 호주로 워킹홀리데이를 떠남"""

ss.setdefault("view", "front")
ss.setdefault("sel", {"preset": 0, "msg": "parcel_smishing"})
ss.setdefault("nonce", 0)
ss.setdefault("custom", None)
ss.setdefault("memo", SAMPLE_MEMO)
ss.setdefault("tp_null", False)
ss.setdefault("personal_days", 0)
for v in SURVEY:
    ss.setdefault(f"pf_{v}", int(DEMO_PROFILE[v]))


def apply_preset(i):
    p = PRESETS[i]
    ss.update(tp=p["tp"], stress=p["stress"], sleep=p["sleep"], wl=p["wl"], events=p["events"],
              tp_null=False, personal_days=0)


if "tp" not in ss:
    apply_preset(0)


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


def context_for(events_text):
    """일정 -> (맥락 점수, 이유, 출처, 맥락 키). 프리셋 일정은 규칙으로 판단해 앞면·뒷면 숫자를 맞춘다."""
    ev = [e for e in events_text.splitlines() if e.strip()]
    if events_text in [p["events"] for p in PRESETS]:
        sc, keys = rule_scores(ev)
        from context_llm import rule_reasons
        return sc, (rule_reasons(ev) if ev else {}), "rule", keys
    with st.spinner("AI 비서가 요즘 일정을 읽는 중…"):
        sc, reasons, src, llm_ctx = llm_scores(ev)
    _, rule_keys = rule_scores(ev)
    return sc, reasons, src, sorted(set(rule_keys) | set(llm_ctx))


def level_of(m):
    return next(name for cut, name in LEVELS if m < cut)


def josa(word, with_batchim, without):
    """마지막 글자 받침에 따라 조사 선택 (예: 스미싱'이 / 안내'가)."""
    ch = word.strip()[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    return without


def explain(drivers, med, tp, stress, sleep):
    """오늘의 설명 문장 — 실제로 '몰렸다·부족하다'고 말할 수 있을 때만 상태를 문장에 넣는다."""
    if not drivers or med < 1.1:
        return "평소와 비슷한 날이에요. 기본 표시만 유지할게요."
    phrases = {"시간 압박": (tp >= 0.5, "일정이 몰려 있고", "일정이 몰려 있어서"),
               "스트레스": (stress >= 1, "스트레스가 평소보다 높고", "스트레스가 평소보다 높아서"),
               "수면 부족": (sleep >= 0.5, "잠이 부족하고", "잠이 부족해서")}
    ctx = [k[4:] for k, _ in drivers if k.startswith("맥락: ")]
    states = [k for k, _ in drivers if k in phrases and phrases[k][0]][:2]
    out = []
    if ctx:
        out.append(f"요즘 일정 때문에 '{ctx[0]}'{josa(ctx[0], '이', '가')} 진짜처럼 느껴지기 쉬워요.")
    if states:
        parts = [phrases[k][1] for k in states[:-1]] + [phrases[states[-1]][2]]
        out.append(f"오늘은 {' '.join(parts)} 서두르기 쉬운 날이에요.")
    return " ".join(out) + " 링크와 송금 요청은 한 번 더 확인하세요."


def current_profile():
    return {v: ss[f"pf_{v}"] for v in SURVEY}


# ---------------------------------------------------------------- 앞면 <-> 뒷면 전환
def on_flip():
    v = ss["stage"].flip or {}
    i = int(v.get("preset", 0))
    ss.sel = {"preset": i, "msg": v.get("msg")}
    apply_preset(i)
    ss.view = "back"


def on_custom():
    v = ss["stage"].custom or {}
    text = str(v.get("text", "")).strip()[:300]
    if not text:
        return
    key, why, src = classify_message(text)
    ss.custom = {"text": text, "key": key, "name": S.SCENARIOS[key]["name"] if key else None,
                 "reason": why, "src": src}
    ss.sel = {"preset": int(v.get("preset", 0)), "msg": "custom"}
    ss.nonce += 1


def go_front():
    ss.view = "front"
    ss.flip_in = True
    ss.nonce += 1


st.markdown("<div class='brand'><b>🛡️ 오늘의 피싱 예보</b>"
            "<span>상태 적응형 사회공학 디지털 트윈 · 시제품은 가상 사용자와 가상 기록만 사용합니다</span></div>",
            unsafe_allow_html=True)

profile = current_profile()
profile_t = tuple(sorted(profile.items()))

# ================================================================ 앞면
if ss.view == "front":
    presets_out = []
    for p in PRESETS:
        sc, _, _, keys = context_for(p["events"])
        r = run_sim(profile_t, p["tp"], p["stress"], p["sleep"], p["wl"], tuple(keys),
                    tuple(sorted(sc.items())), False, 0)
        presets_out.append({
            "label": p["label"], "sub": p["sub"], "today": r["ratio_median"], "level": r["level"]["level"],
            "explain": explain(r["drivers"], r["ratio_median"], p["tp"], p["stress"], p["sleep"]),
            "scen": {k: {"mult": s["ratio_median"], "lo": s["ratio_90"][0], "hi": s["ratio_90"][1]}
                     for k, s in r["scenarios"].items()},
        })
    levels = [{"cut": cut if cut != float("inf") else 1e9, "name": name, "color": LEVEL_STYLE[name][0],
               "bg": LEVEL_STYLE[name][1], "emoji": LEVEL_STYLE[name][2], "weather": LEVEL_STYLE[name][3],
               "action": LEVEL_STYLE[name][4]} for cut, name in LEVELS]
    data = {
        "presets": presets_out, "levels": levels,
        "cards": [{"key": c["key"], "icon": c["icon"], "title": c["title"], "from": c["from_"], "text": c["text"]}
                  for c in CARDS],
        "names": {k: v["name"] for k, v in S.SCENARIOS.items()},
        "sel": ss.sel, "nonce": ss.nonce, "custom": ss.custom,
        "flipIn": bool(ss.pop("flip_in", False)),
    }
    stage(data, on_flip, on_custom)
    st.stop()

# ================================================================ 뒷면 (근거 화면)
tp, stress, sleep, wl = ss.tp, ss.stress, ss.sleep, ss.wl
events = ss.events
sel_i = ss.sel["preset"]
pp = PRESETS[sel_i]
adjusted = (tp, stress, sleep, wl, events) != (pp["tp"], pp["stress"], pp["sleep"], pp["wl"], pp["events"]) \
    or ss.tp_null or ss.personal_days or profile != {v: int(DEMO_PROFILE[v]) for v in SURVEY}

scores, reasons, score_src, ctx_keys = context_for(events)
event_list = [e for e in events.splitlines() if e.strip()]
rule_sc, _ = rule_scores(event_list)
scores_t = tuple(sorted(scores.items()))
args = (profile_t, tp, stress, sleep, wl, tuple(ctx_keys), scores_t)
res = run_sim(*args, ss.tp_null, ss.personal_days)
level = res["level"]["level"]
color = LEVEL_COLOR[level]
med, (lo, hi) = res["ratio_median"], res["ratio_90"]
model = get_model()
mean_profile = {v: model["mu"][j] for j, v in enumerate(VARS)}
base_or = float(np.exp(base_logit(profile, model) - base_logit(mean_profile, model)))
base_p = float(expit(base_logit(profile, model)))

with st.container(key="back"):
    h1, h2 = st.columns([1, 4], vertical_alignment="center")
    h1.button("← 폰으로 돌아가기", on_click=go_front, type="primary", width="stretch")
    h2.markdown(f"#### {pp['label']}{' · 직접 조정한 값' if adjusted else ''} — 왜 이렇게 나왔나")

    emo, wname = LEVEL_STYLE[level][2], LEVEL_STYLE[level][3]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("오늘 취약도 (평소 대비)", f"{med:.2f}배", help="모든 공격 유형 평균")
    m2.metric("90% 범위", f"{lo:.2f}~{hi:.2f}배")
    m3.metric("방어 단계", f"{emo} {level}", help=res["level"]["reason"])
    m4.metric("기본 취약도 (고정층)", f"{base_or:.2f}배", help="같은 조건의 평균 사용자 대비 속을 오즈 (Spamley 기반)")
    st.markdown(f"<div style='border-left:5px solid {color};background:{LEVEL_STYLE[level][1]};padding:12px 16px;"
                f"border-radius:0 10px 10px 0;font-size:16px;margin:4px 0 8px'>"
                f"{explain(res['drivers'], med, tp, stress, sleep)}</div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📊 시뮬레이션 결과", "📚 근거·민감도", "🔄 개인화(동기화)",
                                  "🤖 AI 비서 연동", "🎛️ 직접 조정해 보기"])

    # ------------------------------------------------ 시뮬레이션 결과
    with t1:
        cA, cB = st.columns(2, gap="large")
        with cA:
            st.markdown("**1,000회 반복 시뮬레이션 분포** — 매 회 효과 크기를 근거 범위에서 새로 뽑습니다")
            df = pd.DataFrame({"평소 대비 배수": res["ratio_samples"]})
            hist = alt.Chart(df).mark_bar(color="#5b7bd5", opacity=0.85).encode(
                alt.X("평소 대비 배수:Q", bin=alt.Bin(maxbins=40), title="평소 대비 배수"),
                alt.Y("count()", title="반복 수"))
            rule = alt.Chart(pd.DataFrame({"x": [med]})).mark_rule(color=color, size=3).encode(x="x:Q")
            st.altair_chart((hist + rule).properties(height=260), width="stretch")
            st.caption(f"중앙값 {med:.2f}배 · 90% 범위 {lo:.2f}~{hi:.2f}배. 범위가 넓으면 근거가 약한 요인이 크게 작용한 날입니다.")
        with cB:
            st.markdown("**공격 유형별 평소 대비 배수** (선: 90% 범위)")
            sc = pd.DataFrame([{"공격": s["name"], "중앙값": s["ratio_median"], "하한": s["ratio_90"][0],
                                "상한": s["ratio_90"][1]} for s in res["scenarios"].values()])
            base = alt.Chart(sc).encode(alt.Y("공격:N", sort=alt.EncodingSortField("중앙값", order="descending"),
                                              title=None, axis=alt.Axis(labelLimit=250)))
            st.altair_chart((base.mark_rule(size=2, color="#999").encode(
                x=alt.X("하한:Q", title="평소 대비 배수", scale=alt.Scale(zero=False)), x2="상한:Q")
                + base.mark_point(filled=True, size=140, color="#1f2633").encode(x="중앙값:Q")).properties(height=200),
                width="stretch")
            st.markdown("**오늘 수치를 올린 요인** (속을 오즈 배수, 중앙값)")
            if res["drivers"]:
                st.dataframe(pd.DataFrame(res["drivers"], columns=["요인", "배수"]), hide_index=True, width="stretch")
            else:
                st.write("없음 — 평소와 같은 상태")
        st.caption("요인 배수는 '속을 오즈' 기준이라 곱한 값이 오늘 취약도와 같지 않습니다. 오늘 취약도는 확률 기준으로 바꾼 뒤 "
                   "공격 4종을 평균한 값이고, 맥락 요인은 맞물리는 공격에만 적용됩니다.")
        st.markdown("**방어 단계 규칙**: 중앙값 1.25배 미만 평소 · 1.6배 미만 주의 · 2.2배 미만 경계 · 그 이상 위험. "
                    "대출 준비·이사처럼 잃을 것이 큰 시기에 예측 상한이 다음 단계를 넘으면 한 단계 올립니다. "
                    f"(오늘: {res['level']['reason']})")
        st.markdown(f"**맥락 판단** ({'LLM' if score_src == 'llm' else '키워드 규칙 · 기기 내 경량 판단'})")
        rows = []
        for k, v in scores.items():
            row = {"공격 유형": S.SCENARIOS[k]["name"], "맞물림": round(v, 2)}
            if score_src == "llm":
                row["키워드 규칙이었다면"] = round(rule_sc[k], 2)
            row["이유"] = reasons.get(k, "")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if score_src == "rule" and reasons.get("_error") and not reasons["_error"].startswith("RuntimeError: API 키 없음"):
            st.info("AI 판단 연결이 잠시 원활하지 않아 키워드 규칙으로 판단했어요. 1분 뒤 자동으로 다시 시도합니다.")
        c = ss.custom
        if c:
            st.caption(f"직접 입력한 문자 분류: {c['name'] or '해당 유형 없음'} "
                       f"({'AI 판단' if c['src'] == 'llm' else '키워드 규칙'}) · {c['reason']} — "
                       "문자 자체의 위험도가 아니라 공격 유형만 고르고, 배수는 위 시뮬레이션 값을 씁니다.")

    # ------------------------------------------------ 근거·민감도
    with t2:
        center, srows = run_sensitivity(*args)
        st.markdown(f"**민감도 분석 — 어떤 가정이 결과를 가장 크게 바꾸나** (모든 가정 중앙값일 때 {center:.2f}배)")
        sd = pd.DataFrame(srows)
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
            "요인": {"time_pressure": "시간 압박", "stress": "스트레스", "workload": "업무 부하",
                   "sleep_deficit": "수면 부족"}[k],
            "범위(OR)": f"{v.get('or_low', v.get('context_weight_low'))} ~ {v.get('or_high', v.get('context_weight_high'))}",
            "단위": v["unit"], "근거 강도": v["strength"], "근거": v["basis"], "확인 필요": v.get("todo", "")}
            for k, v in P.STATE_EFFECTS.items()]), hide_index=True, width="stretch")
        st.markdown(f"**맥락 일치 효과**: OR {S.CONTEXT_OR_LOW}~{S.CONTEXT_OR_HIGH} (설계 가정, 수치 근거는 확인한 범위에서 없음)")
        st.markdown(f"**고정층**: Spamley 공개 통계(N={P.SPAMLEY_N}, 평균 탐지율 {P.SPAMLEY_MEAN_RECALL}) 기반 합성 데이터로 학습 · "
                    "변수 12개의 탐지율 상관을 오차 0.01 이내로 재현")
        st.markdown("**한계**: 상태층 효과는 서로 다른 연구의 효과 크기를 조합한 것으로, 조합 자체는 아직 검증 전입니다. "
                    "다른 사람의 실제 상태 데이터를 구할 수 없어 시제품은 가상 사용자로만 시연합니다.")

    # ------------------------------------------------ 개인화
    with t3:
        st.markdown("**현실과의 동기화** — 실제 반응(신고·무시·열어봄)으로 이 사람에게 맞게 값을 고칩니다")
        st.caption("시연 시나리오: 잠이 부족한 날에도 의심 문자를 꾸준히 신고하는 가상 사용자 (하루 평균 1건의 가상 기록)")
        curve, before, after = personalization_curve(*args)
        band = alt.Chart(curve).mark_area(opacity=0.2, color="#5b7bd5").encode(
            x=alt.X("사용 일수:Q"), y=alt.Y("하한:Q", title="오늘 취약도 (평소 대비)", scale=alt.Scale(zero=False)), y2="상한:Q")
        line = alt.Chart(curve).mark_line(point=True, color="#1f2633").encode(x="사용 일수:Q", y="중앙값:Q")
        st.altair_chart(band + line, width="stretch")
        cA, cB = st.columns(2)
        ren = {"factor": "요인", "population_or": "평균 효과", "personal_or": "이 사람", "certainty": "확실도"}
        cA.markdown("**처음 (선행 연구 평균값)**")
        cA.dataframe(pd.DataFrame(before).rename(columns=ren), hide_index=True, width="stretch")
        cB.markdown("**180일 후**")
        cB.dataframe(pd.DataFrame(after).rename(columns=ren), hide_index=True, width="stretch")
        st.caption("한계: 의심 메시지가 드물면 개인화에 수개월이 걸립니다. 반응 기록은 기기 밖으로 나가지 않는다는 전제입니다.")

    # ------------------------------------------------ AI 비서 연동
    def apply_memo():
        """메모 -> LLM이 설문 초안·일정 추출 -> 값에 반영 (사용자가 확인·수정 후 통계 모델이 계산)."""
        try:
            out = llm_profile(ss.get("memo", ""))
        except Exception as e:
            print("트윈 자동 생성 실패 :", type(e).__name__, e, flush=True)
            ss["auto_profile"] = {"error": f"{type(e).__name__}"}
            return
        for k, v in out["survey"].items():
            ss[f"pf_{k}"] = v
        if out["events"]:
            ss["events"] = "\n".join(out["events"])
        ss["auto_profile"] = out

    def load_presenter_memo():
        """발표자 본인 메모는 저장소가 아니라 Streamlit Secrets(PRESENTER_MEMO)에서만 읽는다."""
        try:
            ss["memo"] = st.secrets["PRESENTER_MEMO"]
        except Exception:
            pass

    def reset_profile():
        for v in SURVEY:
            ss[f"pf_{v}"] = int(DEMO_PROFILE[v])
        ss.pop("auto_profile", None)

    with t4:
        st.markdown("**AI 비서 연동 — 이미 아는 정보로 트윈을 자동으로 만듭니다**")
        st.caption("개인 AI 비서가 기억하는 메모를 읽어 고정층 설문 초안과 요즘 일정을 채웁니다. "
                   "LLM은 초안만 만들고, 사용자가 확인·수정한 값으로 통계 모델이 취약도를 계산합니다.")
        st.text_area("AI 비서 메모 (시연은 가상 인물)", key="memo", height=230)
        has_pm = False
        try:
            has_pm = "PRESENTER_MEMO" in st.secrets
        except Exception:
            pass
        if has_pm:
            q1, q2 = st.columns(2)
            q1.button("가상 인물 메모", on_click=lambda: ss.update(memo=SAMPLE_MEMO), width="stretch")
            q2.button("발표자 본인 메모 (동의한 부분만)", on_click=load_presenter_memo, width="stretch")
        b1, b2 = st.columns(2)
        b1.button("트윈 자동 생성", on_click=apply_memo, type="primary", width="stretch")
        b2.button("가상 사용자 기본값으로 되돌리기", on_click=reset_profile, width="stretch", key="reset2")
        auto = ss.get("auto_profile")
        if auto and "error" in auto:
            st.warning("AI 연결이 원활하지 않아 자동 생성을 하지 못했어요. '직접 조정해 보기'에서 설문을 직접 입력할 수 있어요.")
        elif auto:
            st.success(f"설문 {len(auto['survey'])}/{len(SURVEY)}개 항목과 일정 {len(auto['events'])}개를 채웠어요. "
                       "위 수치는 이미 새 트윈으로 다시 계산됐어요.")
            cA, cB = st.columns([1.4, 1])
            with cA:
                st.markdown("**설문 초안**")
                st.dataframe(pd.DataFrame([{
                    "항목": SURVEY[k][0], "추정값": str(auto["survey"].get(k, "—")),
                    "근거": auto["reasons"].get(k, "") if k in auto["survey"] else "메모에 근거 없음 → 기본값 유지"}
                    for k in SURVEY]), hide_index=True, width="stretch")
            with cB:
                st.markdown("**뽑아낸 요즘 일정**")
                for e in auto["events"] or ["없음"]:
                    st.markdown(f"- {e}")
        st.divider()
        st.markdown("**왜 이 값이 나왔나** — 세 층이 각각 무엇을 바꿨는지")
        filled = auto.get("survey", {}) if auto and "survey" in auto else {}
        x_now = np.array([profile[v] for v in VARS], dtype=float)
        parts = model["coef_mean"] * (x_now - model["mu"]) / model["sd"]
        fx = []
        for j, v in enumerate(VARS):
            if abs(parts[j]) < 0.02:
                continue
            o = float(np.exp(parts[j]))
            fx.append((abs(parts[j]), {"항목": SURVEY[v][0] + (" ✦" if v in filled else ""),
                                       "입력값": int(profile[v]), "평균 사용자": round(float(model["mu"][j]), 1),
                                       "영향": f"{o:.2f}배 · {'덜 속는 쪽' if o < 1 else '더 속는 쪽'}"}))
        fx = [r for _, r in sorted(fx, key=lambda t: -t[0])]
        w1, w2, w3 = st.columns(3, gap="medium")
        with w1:
            st.markdown(f"**① 고정층 · 어떤 사람인가**  \n평균 사용자 대비 **{base_or:.2f}배**")
            if fx:
                st.dataframe(pd.DataFrame(fx), hide_index=True, width="stretch")
            st.caption("Spamley 공개 통계로 학습한 로지스틱 회귀 계수 × (입력값 − 평균). "
                       + ("✦ = 메모로 채운 항목, 나머지는 기본값." if filled else ""))
        with w2:
            st.markdown("**② 상태층 · 오늘 컨디션**")
            st_drv = {k: v for k, v in res["drivers"] if not k.startswith("맥락: ")}
            srows2 = [("시간 압박", f"{tp:.1f}"), ("스트레스", f"+{stress}"), ("수면 부족", f"{sleep:.1f}"), ("업무량", f"{wl:.1f}")]
            st.dataframe(pd.DataFrame([{"상태": n, "입력값": val,
                                        "속을 오즈": f"{st_drv[n]:.2f}배" if n in st_drv else
                                        ("맥락 효과에 반영" if n == "업무량" and any(v >= 0.5 for v in scores.values())
                                         else "영향 없음")} for n, val in srows2]), hide_index=True, width="stretch")
            st.caption("목표 서비스는 캘린더 일정 밀도·수면 기록·하루 한 번 스트레스 체크에서 받습니다.")
        with w3:
            st.markdown("**③ 맥락층 · 요즘 일정**")
            hit = [(S.SCENARIOS[k]["name"], v, reasons.get(k, "")) for k, v in scores.items() if v >= 0.5]
            if hit:
                for name, v, why in sorted(hit, key=lambda t: -t[1]):
                    st.markdown(f"- **{name}** (맞물림 {v:.1f})" + (f"  \n  {why}" if why else ""))
            else:
                st.markdown("- 맞물리는 공격 없음 → 맥락 효과 0")
            st.caption("LLM은 일정과 공격 유형이 얼마나 맞물리는지만 0~1로 판단하고, 확률 계산에는 관여하지 않습니다.")
        st.caption("※ 시연은 가상 인물 메모만 사용합니다. 목표 서비스에서는 사용자의 AI 비서가 기기 안에서 처리하고, "
                   "메모 원문이 아닌 설문 초안만 트윈에 넘깁니다. 메모로 추정한 값이 실제 설문과 얼마나 맞는지는 아직 검증 전입니다.")

    # ------------------------------------------------ 직접 조정
    with t5:
        st.caption("값을 바꾸면 이 화면의 수치가 바로 다시 계산돼요. 폰으로 돌아가면 상태 버튼 값으로 돌아가요.")
        a1, a2 = st.columns(2, gap="large")
        with a1:
            st.markdown("**오늘의 상태**")
            st.slider("시간 압박 (캘린더 일정 밀도)", 0.0, 1.0, key="tp", step=0.1)
            st.slider("스트레스 (평소보다 몇 점 높은가)", 0, 4, key="stress")
            st.slider("수면 부족 (1 = 5시간 이하)", 0.0, 1.0, key="sleep", step=0.1)
            st.slider("업무량", 0.0, 1.0, key="wl", step=0.1)
            st.markdown("**요즘 일정 (맥락)** — 프리셋 외 문장은 LLM이 읽어요")
            st.text_area("한 줄에 하나씩", key="events", height=90)
            st.markdown("**옵션**")
            st.toggle("시간 압박 효과 없음 가정 (Black & Sarno 2023)", key="tp_null")
            st.select_slider("개인화: 반응 기록 사용 일수", [0, 30, 90, 180], key="personal_days")
        with a2:
            st.markdown("**고정층 설문 (가상 사용자)**")
            s1, s2 = st.columns(2)
            for i, (v, (label, lo_, hi_, step)) in enumerate(SURVEY.items()):
                (s1 if i % 2 == 0 else s2).slider(label, lo_, hi_, step=step, key=f"pf_{v}")
            st.button("가상 사용자 기본값으로", on_click=reset_profile, width="stretch")
