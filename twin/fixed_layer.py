"""
고정층: Spamley 공개 통계로 합성 사용자를 만들고, 로지스틱 회귀로 '기본 취약도' 모델을 학습한다.

절차
 1) 가우시안 코퓰라로 변수 12개 + 탐지 성향(잠재값)을 함께 생성
 2) 각 변수의 주변분포를 논문 표의 평균·표준편차·범위에 맞춤
 3) 탐지 성향 -> 사용자별 탐지 확률 -> Spamley 시험 구조(10통 중 피싱 n통)로 탐지 결과 생성
 4) 생성된 탐지율과 각 변수의 Spearman 상관이 논문 값과 같아지도록 잠재 상관을 반복 보정
 5) 원 표본 크기(N=1086)로 여러 번 합성 -> 매번 로지스틱 회귀 -> 계수의 평균과 흩어짐을 저장
    (흩어짐은 시뮬레이션에서 고정층 불확실성으로 사용)

주의: 원자료가 아니라 '공개된 주변 통계'만 맞춘 합성 데이터다.
변수 간 상관(나이-경력 제외)과 탐지 확률의 개인차 크기(LATENT_SCALE)는 설계 가정이다.
"""
import json
import numpy as np
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

import params as P

VARS = list(P.SPAMLEY_VARS.keys())
K = len(VARS)
LATENT_SCALE = 1.0   # 설계 가정: 사용자 간 탐지 확률 차이의 크기 (logit 척도 표준편차)


# ---------------------------------------------------------------- 주변분포 변환
def _to_marginal(z, spec):
    kind, mean, sd, lo, hi, _ = spec
    if kind == "binary":
        return (z > stats.norm.ppf(1 - mean)).astype(float)
    if kind == "likert":
        return np.clip(np.round(mean + sd * z), lo, hi)
    # cont: 하한에서 시작하는 감마분포로 오른쪽 꼬리(나이·경력) 표현
    m = mean - lo
    shape, scale = (m / sd) ** 2, sd ** 2 / m
    x = lo + stats.gamma.ppf(stats.norm.cdf(z), shape, scale=scale)
    return np.clip(x, lo, hi)


def _corr_matrix(r_target):
    """변수 간 상관(가정) + 변수-탐지성향 상관(보정 대상)으로 전체 상관행렬 구성."""
    C = np.eye(K + 1)
    for (a, b), r in P.SPAMLEY_INTERCORR.items():
        i, j = VARS.index(a), VARS.index(b)
        C[i, j] = C[j, i] = r
    C[:K, K] = C[K, :K] = r_target
    # 양정치 보정 (필요할 때만)
    w, v = np.linalg.eigh(C)
    if w.min() < 1e-6:
        C = v @ np.diag(np.maximum(w, 1e-6)) @ v.T
        d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    return C


# ---------------------------------------------------------------- 합성
def synthesize(n, r_latent, intercept, rng):
    C = _corr_matrix(r_latent)
    Z = rng.multivariate_normal(np.zeros(K + 1), C, size=n)
    X = np.column_stack([_to_marginal(Z[:, j], P.SPAMLEY_VARS[v]) for j, v in enumerate(VARS)])
    p_detect = expit(intercept + LATENT_SCALE * Z[:, K])

    # Spamley 시험 구조: 10통 중 피싱 수는 무작위, 3통 미만은 제외
    n_phish = rng.binomial(P.EMAILS_PER_USER, 0.5, size=n)
    keep = n_phish >= P.MIN_PHISHING
    X, p_detect, n_phish = X[keep], p_detect[keep], n_phish[keep]
    hits = rng.binomial(n_phish, p_detect)
    return X, n_phish, hits


def _observed_rho(X, n_phish, hits):
    recall = hits / n_phish
    return np.array([stats.spearmanr(X[:, j], recall).statistic for j in range(K)])


def calibrate(rng, n_big=200_000, iters=8, verbose=True):
    """잠재 상관과 절편을 보정해 논문의 Spearman 상관과 평균 탐지율을 재현."""
    target = np.array([P.SPAMLEY_VARS[v][5] for v in VARS])
    r = 2 * np.sin(np.pi * target / 6)          # Spearman -> Pearson 초기값
    a = logit(P.SPAMLEY_MEAN_RECALL)
    for it in range(iters):
        X, n_ph, hits = synthesize(n_big, r, a, rng)
        obs = _observed_rho(X, n_ph, hits)
        mean_recall = (hits / n_ph).mean()
        r = np.clip(r * np.where(np.abs(obs) > 1e-4, target / obs, 1.0), -0.95, 0.95)
        a += logit(P.SPAMLEY_MEAN_RECALL) - logit(mean_recall)
        if verbose:
            err = np.abs(obs - target).max()
            print(f"  보정 {it+1}: 최대 상관 오차 {err:.4f}, 평균 탐지율 {mean_recall:.3f}")
    return r, a


# ---------------------------------------------------------------- 학습
def fit_once(X, n_phish, hits, mu, sd):
    """사용자별 결과를 메일 단위 행으로 펼쳐 '속음(놓침)' 확률을 학습."""
    Xs = (X - mu) / sd
    rows_X = np.repeat(Xs, 2, axis=0)
    y = np.tile([1, 0], len(X))                       # 1 = 속음(놓침), 0 = 탐지
    w = np.column_stack([n_phish - hits, hits]).ravel()
    m = LogisticRegression(C=1e6, max_iter=2000)
    m.fit(rows_X, y, sample_weight=w)
    return m.intercept_[0], m.coef_[0]


def train(r_latent, intercept, rng, n_draws=200):
    # 표준화 기준은 대표본에서 한 번 계산 (트윈 입력도 같은 기준 사용)
    Xb, _, _ = synthesize(200_000, r_latent, intercept, rng)
    mu, sd = Xb.mean(0), Xb.std(0)
    b0s, Bs = [], []
    for _ in range(n_draws):
        X, n_ph, hits = synthesize(P.SPAMLEY_N, r_latent, intercept, rng)
        b0, B = fit_once(X, n_ph, hits, mu, sd)
        b0s.append(b0); Bs.append(B)
    return mu, sd, np.array(b0s), np.array(Bs)


# ---------------------------------------------------------------- 검증
def validate(r_latent, intercept, model, rng):
    mu, sd, b0, B = model["mu"], model["sd"], model["b0_mean"], model["coef_mean"]
    target = np.array([P.SPAMLEY_VARS[v][5] for v in VARS])
    out = {}

    # (1) 원 표본 크기에서 상관 재현도 (200회 평균)
    rhos = []
    for _ in range(200):
        X, n_ph, hits = synthesize(P.SPAMLEY_N, r_latent, intercept, rng)
        rhos.append(_observed_rho(X, n_ph, hits))
    rhos = np.array(rhos)
    out["rho_table"] = [
        {"var": v, "paper": float(target[j]), "synthetic_mean": float(rhos[:, j].mean()),
         "synthetic_95": [float(np.percentile(rhos[:, j], 2.5)), float(np.percentile(rhos[:, j], 97.5))]}
        for j, v in enumerate(VARS)]

    # (2) 집단 차이 재현 (근사 검증)
    #  논문은 연차(Seniority)·창의성(Creativity) 요인으로 군집화. 창의성 요인 구성 변수(개방성·외향성 등)는
    #  모델에 없으므로, 연차 합성점수(나이·경력·학력) 상위 28.6%를 '인식군'의 근사로 사용.
    X, n_ph, hits = synthesize(200_000, r_latent, intercept, rng)
    recall = hits / n_ph
    sen = ((X[:, [0, 1, 3]] - mu[[0, 1, 3]]) / sd[[0, 1, 3]]).mean(1)
    cut = np.quantile(sen, 1 - P.CLUSTER_AWARE["share"])
    out["cluster_check"] = {
        "aware_proxy_recall": float(recall[sen >= cut].mean()),
        "highrisk_proxy_recall": float(recall[sen < cut].mean()),
        "paper_aware_recall": P.CLUSTER_AWARE["recall"],
        "paper_highrisk_recall": P.CLUSTER_HIGH_RISK["recall"],
    }

    # (3) 새 합성 표본에서 예측-관측 일치 (내부 일관성 점검, 실제 타당도 아님)
    X, n_ph, hits = synthesize(P.SPAMLEY_N, r_latent, intercept, rng)
    pred_fall = expit(b0 + ((X - mu) / sd) @ B)
    out["holdout_spearman_pred_vs_obs"] = float(stats.spearmanr(1 - pred_fall, hits / n_ph).statistic)
    out["holdout_mean_pred_recall"] = float((1 - pred_fall).mean())
    out["holdout_mean_obs_recall"] = float((hits / n_ph).mean())
    return out


# ---------------------------------------------------------------- 트윈용 함수
def base_logit(profile: dict, model: dict, coef=None, b0=None):
    """설문 응답(dict) -> 고정층 기본 '속을 오즈'(logit). coef/b0를 넘기면 그 표본값 사용."""
    x = np.array([profile.get(v, model["mu"][j]) for j, v in enumerate(VARS)])
    xs = (x - model["mu"]) / model["sd"]
    coef = model["coef_mean"] if coef is None else coef
    b0 = model["b0_mean"] if b0 is None else b0
    return float(b0 + xs @ coef)


def main():
    rng = np.random.default_rng(20261028)
    print("1) 잠재 상관 보정")
    r, a = calibrate(rng)
    print("2) 원 표본 크기로 200회 합성·학습")
    mu, sd, b0s, Bs = train(r, a, rng)
    model = {
        "vars": VARS, "mu": mu, "sd": sd,
        "b0_mean": b0s.mean(), "b0_sd": b0s.std(),
        "coef_mean": Bs.mean(0), "coef_sd": Bs.std(0), "coef_draws": Bs, "b0_draws": b0s,
        "latent_r": r, "latent_intercept": a, "latent_scale": LATENT_SCALE,
    }
    print("3) 검증")
    val = validate(r, a, model, rng)

    def js(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating,)): return float(o)
        return o
    with open("fixed_layer_model.json", "w", encoding="utf-8") as f:
        json.dump({k: js(v) for k, v in model.items()}, f, ensure_ascii=False)
    with open("fixed_layer_validation.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print("\n[상관 재현]  변수 | 논문 rho | 합성 평균 (95% 범위)")
    for row in val["rho_table"]:
        lo, hi = row["synthetic_95"]
        print(f"  {row['var']:<18} {row['paper']:+.3f} | {row['synthetic_mean']:+.3f} ({lo:+.3f} ~ {hi:+.3f})")
    c = val["cluster_check"]
    print(f"\n[집단 차이] 인식군 근사 {c['aware_proxy_recall']:.3f} (논문 {c['paper_aware_recall']}) | "
          f"고위험군 근사 {c['highrisk_proxy_recall']:.3f} (논문 {c['paper_highrisk_recall']})")
    print(f"[홀드아웃] 예측-관측 Spearman {val['holdout_spearman_pred_vs_obs']:.3f}, "
          f"평균 탐지율 예측 {val['holdout_mean_pred_recall']:.3f} / 관측 {val['holdout_mean_obs_recall']:.3f}")
    print("\n[계수] 표준화 1SD당 '속을 오즈' 배수 (평균, 200회 표본의 95% 범위)")
    for j, v in enumerate(VARS):
        lo, hi = np.percentile(Bs[:, j], [2.5, 97.5])
        print(f"  {v:<18} x{np.exp(Bs[:, j].mean()):.2f} ({np.exp(lo):.2f} ~ {np.exp(hi):.2f})")


if __name__ == "__main__":
    main()
