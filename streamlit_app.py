# -*- coding: utf-8 -*-
"""
MIX — Concrete Strength Studio
머신러닝(Gradient Boosting) 기반 콘크리트 압축강도 예측 웹앱.

입력 4종 (학습 시 컬럼 순서와 동일):
    ['시멘트량', '고성능 감수제량', '재령 기간', '물양']
출력: 콘크리트 압축강도 (MPa)

디자인: mdx.so 의 라이트 미니멀 무드 + 한글 웹폰트(페이퍼로지 / 서울알림체).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (반드시 첫 st 호출)
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIX · Concrete Strength Studio",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 학습 시 X 컬럼 순서 — 절대 바꾸지 말 것 (모델 feature_names_in_ 과 일치해야 함)
FEATURES = ["시멘트량", "고성능 감수제량", "재령 기간", "물양"]
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "concrete_strength_new_model.pkl"
DATA_PATH = BASE_DIR / "model_data.npz"


@st.cache_resource(show_spinner=False)
def get_model():
    """원본 .pkl을 먼저 시도하고, 환경(파이썬·scikit-learn 버전) 때문에 못 불러오면
    동봉한 distilled 데이터(model_data.npz)로 동일한 모델을 즉석 재학습한다.
    → 어떤 배포 환경에서도 절대 죽지 않게 만드는 안전장치."""
    # 1) 원본 모델 그대로 사용 (scikit-learn 버전이 맞는 환경에서)
    try:
        import joblib

        m = joblib.load(MODEL_PATH)
        m.predict(pd.DataFrame([[300, 5, 28, 180]], columns=FEATURES))  # 동작 확인
        return m, "원본 모델 (concrete_strength_new_model.pkl)"
    except Exception:
        pass

    # 2) 폴백 — 원본 모델의 예측을 학습한 distilled 데이터로 재학습 (버전 무관)
    from sklearn.ensemble import GradientBoostingRegressor

    arr = np.load(DATA_PATH)["data"].astype(float)
    X = pd.DataFrame(arr[:, :4], columns=FEATURES)
    y = arr[:, 4]
    m = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.9, random_state=0,
    ).fit(X, y)
    return m, "호환 재학습 모델 (Gradient Boosting)"


try:
    model, MODEL_SOURCE = get_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"모델을 준비하지 못했습니다.\n\n{exc}")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ---------- Web fonts (눈누 / jsDelivr) ---------- */
@font-face{font-family:'Paperlogy';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-3@1.0/Paperlogy-3Light.woff2') format('woff2');font-weight:300;font-display:swap;}
@font-face{font-family:'Paperlogy';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-3@1.0/Paperlogy-4Regular.woff2') format('woff2');font-weight:400;font-display:swap;}
@font-face{font-family:'Paperlogy';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-3@1.0/Paperlogy-5Medium.woff2') format('woff2');font-weight:500;font-display:swap;}
@font-face{font-family:'Paperlogy';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-3@1.0/Paperlogy-6SemiBold.woff2') format('woff2');font-weight:600;font-display:swap;}
@font-face{font-family:'Paperlogy';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-3@1.0/Paperlogy-7Bold.woff2') format('woff2');font-weight:700;font-display:swap;}
@font-face{font-family:'SeoulAlrim';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2505-1@1.0/SeoulAlrimTTF-Medium.woff2') format('woff2');font-weight:500;font-display:swap;}
@font-face{font-family:'SeoulAlrim';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2505-1@1.0/SeoulAlrimTTF-Bold.woff2') format('woff2');font-weight:700;font-display:swap;}
@font-face{font-family:'SeoulAlrim';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/2505-1@1.0/SeoulAlrimTTF-Heavy.woff2') format('woff2');font-weight:900;font-display:swap;}

:root{
  --ink:#0B0B0C;
  --muted:#9b9a96;
  --muted-2:#bdbcb7;
  --line:rgba(11,11,12,.12);
  --line-soft:rgba(11,11,12,.07);
  --accent:#FF5B23;
  --accent-soft:#ffe7da;
  --glass:rgba(255,255,255,.46);
  --glass-brd:rgba(255,255,255,.7);
  --bg-0:#f3f2ef;
  --bg-1:#e7e6e2;
  --bg-2:#dedcd7;
}

/* ---------- App canvas ---------- */
html,body,[class*="css"], .stApp, [data-testid="stAppViewContainer"]{
  font-family:'Paperlogy',-apple-system,BlinkMacSystemFont,sans-serif;
  color:var(--ink);
  letter-spacing:-.01em;
}
.stApp{
  background:
    radial-gradient(135% 110% at 50% -8%, #f7f6f3 0%, var(--bg-0) 38%, var(--bg-1) 74%, var(--bg-2) 100%);
}
/* film grain */
.stApp::before{
  content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.05;mix-blend-mode:multiply;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
/* hide streamlit chrome */
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer{display:none!important;}
[data-testid="stDecoration"]{display:none!important;}
.block-container, [data-testid="stMainBlockContainer"]{
  padding:0 clamp(20px,4.5vw,76px) 70px!important;max-width:1480px!important;
}
[data-testid="stAppViewContainer"] > .main{overflow:visible;}

/* ---------- top bar ---------- */
.topbar{display:flex;align-items:center;justify-content:space-between;
  padding:30px 2px 0;position:relative;z-index:3;
  animation:rise .9s cubic-bezier(.2,.7,.2,1) both;}
.brand{font-family:'SeoulAlrim';font-weight:900;font-size:25px;letter-spacing:.42em;
  display:flex;align-items:center;gap:14px;}
.brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 5px var(--accent-soft);margin-left:-2px;}

/* ---------- hero ---------- */
.hero{position:relative;z-index:2;min-height:min(86vh,820px);
  display:grid;grid-template-columns:1fr;align-items:end;padding-top:6px;}
.orb-wrap{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  pointer-events:none;z-index:1;}
.orb{width:min(46vw,520px);aspect-ratio:1;animation:floaty 9s ease-in-out infinite;}
.hero-grid{position:relative;z-index:2;width:100%;
  display:grid;grid-template-columns:1.15fr .85fr;gap:30px;align-items:end;
  padding-bottom:14px;}
.eyebrow{font-size:13px;letter-spacing:.26em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin-bottom:22px;animation:rise 1s .05s both;}
.eyebrow b{color:var(--accent);}
.h-title{font-family:'Paperlogy';font-weight:600;line-height:.96;letter-spacing:-.035em;
  font-size:clamp(46px,7vw,104px);margin:0;}
.h-title .l1{animation:rise 1s .12s both;}
.h-title .l2{display:block;color:#46443f;animation:rise 1s .22s both;}
.h-title .l2 em{font-style:normal;color:var(--accent);}
.h-sub{max-width:540px;margin-top:26px;color:#6b6a66;font-weight:400;font-size:clamp(15px,1.3vw,18px);
  line-height:1.62;animation:rise 1s .32s both;}
.cta{margin-top:34px;display:inline-flex;align-items:center;gap:12px;
  background:var(--ink);border-radius:40px;padding:17px 30px;
  font-weight:600;font-size:15.5px;letter-spacing:.02em;cursor:pointer;
  box-shadow:0 18px 38px -18px rgba(0,0,0,.55);transition:transform .35s,box-shadow .35s;
  animation:rise 1s .42s both;}
.cta, .cta:link, .cta:visited, .cta:hover, .cta:active{
  color:#fff!important;text-decoration:none!important;}
.cta:hover{transform:translateY(-3px);box-shadow:0 26px 46px -18px rgba(0,0,0,.6);}
.cta .ar{color:var(--accent)!important;font-weight:700;}

.hero-right{align-self:end;display:flex;flex-direction:column;gap:22px;
  animation:rise 1s .5s both;}
.hr-copy{font-size:clamp(15px,1.25vw,18px);line-height:1.62;color:#4a4945;font-weight:500;}
.hr-copy b{color:var(--ink);font-weight:700;}
.tags{display:flex;flex-wrap:wrap;gap:9px;}
.tag{border:1px solid var(--line);border-radius:30px;padding:9px 17px;font-size:13px;
  font-weight:500;color:#3a3a3c;background:rgba(255,255,255,.4);backdrop-filter:blur(6px);
  transition:all .3s;}
.tag:hover{border-color:var(--ink);transform:translateY(-2px);}
.tag.plus{color:var(--accent);border-color:var(--accent-soft);}

/* ---------- section heading ---------- */
.sec{margin:18px 2px 6px;display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap;}
.sec-k{font-size:12.5px;letter-spacing:.26em;text-transform:uppercase;color:var(--muted);font-weight:600;}
.sec-t{font-family:'Paperlogy';font-weight:600;font-size:clamp(28px,3.4vw,46px);line-height:1.04;
  letter-spacing:-.03em;margin:10px 0 0;}
.sec-t em{font-style:normal;color:var(--accent);}
.sec-d{max-width:430px;color:#6b6a66;font-size:14.5px;line-height:1.6;font-weight:400;}

/* ---------- panels (columns -> glass cards) ---------- */
[data-testid="stColumn"]{
  background:var(--glass);border:1px solid var(--glass-brd);border-radius:26px;
  padding:34px 34px 26px!important;backdrop-filter:blur(16px) saturate(1.1);
  box-shadow:0 30px 60px -40px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.7);
}
.panel-h{display:flex;align-items:center;gap:12px;margin-bottom:6px;}
.panel-h .idx{font-family:'SeoulAlrim';font-weight:900;color:var(--accent);font-size:15px;}
.panel-h .t{font-family:'Paperlogy';font-weight:600;font-size:19px;letter-spacing:-.02em;}
.panel-h .s{color:var(--muted);font-size:13px;font-weight:400;margin-left:auto;}

/* input row label */
.ic{display:flex;align-items:flex-end;justify-content:space-between;margin:20px 0 6px;}
.ic-l{display:flex;flex-direction:column;gap:2px;}
.ic-kr{font-family:'Paperlogy';font-weight:600;font-size:16px;letter-spacing:-.02em;}
.ic-en{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:500;}
.ic-rng{font-family:'Paperlogy';font-size:11.5px;color:var(--muted-2);font-weight:500;letter-spacing:.02em;}

/* ---------- number inputs (키보드 입력 가능) ---------- */
[data-testid="stNumberInput"]{margin:0 0 4px;}
[data-testid="stNumberInput"] [data-baseweb="input"]{
  background:rgba(255,255,255,.6)!important;border:1.5px solid rgba(11,11,12,.1)!important;
  border-radius:15px!important;overflow:hidden;transition:border-color .2s,box-shadow .2s;}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within{
  border-color:var(--accent)!important;box-shadow:0 0 0 4px var(--accent-soft)!important;}
[data-testid="stNumberInput"] input{
  font-family:'SeoulAlrim'!important;font-weight:700!important;font-size:25px!important;
  color:var(--ink)!important;background:transparent!important;padding:9px 14px!important;
  -webkit-text-fill-color:var(--ink)!important;}
[data-testid="stNumberInput"] button{
  background:transparent!important;border:none!important;color:var(--muted)!important;
  border-left:1px solid rgba(11,11,12,.07)!important;transition:color .2s;}
[data-testid="stNumberInput"] button:hover{color:var(--accent)!important;background:var(--accent-soft)!important;}
[data-testid="stNumberInput"] button svg{fill:currentColor!important;}

/* ---------- result card ---------- */
.res-lab{font-size:12.5px;letter-spacing:.24em;text-transform:uppercase;color:var(--muted);font-weight:600;}
.res-big{font-family:'SeoulAlrim';font-weight:900;line-height:.92;letter-spacing:-.02em;
  font-size:clamp(74px,9vw,112px);margin:8px 0 0;
  background:linear-gradient(180deg,#171717 0%,#3b3a37 100%);-webkit-background-clip:text;
  background-clip:text;-webkit-text-fill-color:transparent;
  animation:pop .6s cubic-bezier(.2,.8,.2,1) both;}
.res-big .u{font-family:'Paperlogy';font-weight:600;font-size:.26em;color:var(--accent);
  -webkit-text-fill-color:var(--accent);letter-spacing:0;margin-left:6px;}
.res-grade{display:inline-flex;align-items:center;gap:9px;margin-top:14px;
  background:var(--accent-soft);color:#c2410c;border-radius:30px;padding:8px 16px;
  font-weight:600;font-size:14px;animation:rise .6s .1s both;}
.res-grade .pip{width:7px;height:7px;border-radius:50%;background:var(--accent);}
.res-note{margin-top:14px;color:#6b6a66;font-size:14px;line-height:1.6;font-weight:400;}

/* gauge */
.gauge{margin-top:26px;}
.gauge-bar{position:relative;height:9px;border-radius:30px;
  background:linear-gradient(90deg,#e5e4e0 0%,#f3c9b4 45%,var(--accent) 100%);overflow:visible;}
.gauge-mk{position:absolute;top:50%;width:18px;height:18px;border-radius:50%;background:#fff;
  border:3px solid var(--accent);transform:translate(-50%,-50%);
  box-shadow:0 4px 12px -2px rgba(255,91,35,.6);transition:left .5s cubic-bezier(.2,.8,.2,1);}
.gauge-sc{display:flex;justify-content:space-between;margin-top:10px;color:var(--muted);
  font-size:11.5px;font-weight:500;letter-spacing:.04em;}
.feat-mini{display:flex;justify-content:space-between;gap:8px;margin-top:24px;
  border-top:1px solid var(--line-soft);padding-top:18px;}
.fm{flex:1;text-align:center;}
.fm .k{font-size:11px;color:var(--muted);font-weight:500;letter-spacing:.04em;margin-bottom:3px;}
.fm .v{font-family:'SeoulAlrim';font-weight:700;font-size:17px;}
.fm + .fm{border-left:1px solid var(--line-soft);}

/* ---------- footer ---------- */
.foot{margin:46px 2px 0;padding-top:26px;border-top:1px solid var(--line);
  display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap;
  color:var(--muted);font-size:13px;font-weight:400;line-height:1.7;}
.foot b{color:#3a3a3c;font-weight:600;}
.foot .r{text-align:right;}

/* ---------- keyframes ---------- */
@keyframes rise{from{opacity:0;transform:translateY(26px);}to{opacity:1;transform:translateY(0);}}
@keyframes pop{from{opacity:0;transform:translateY(14px) scale(.96);}to{opacity:1;transform:translateY(0) scale(1);}}
@keyframes floaty{0%,100%{transform:translateY(0) rotate(0);}50%{transform:translateY(-26px) rotate(2.5deg);}}

@media(max-width:900px){
  .hero-grid{grid-template-columns:1fr;gap:26px;}
  .hero-right{margin-top:8px;}
  .orb{width:74vw;opacity:.9;}
  .feat-mini{flex-wrap:wrap;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
#  HERO  (정적 — 분위기/브랜딩)
# ────────────────────────────────────────────────────────────────────────────
ORB_SVG = """
<svg class="orb" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <radialGradient id="sphere" cx="40%" cy="34%" r="68%">
    <stop offset="0%" stop-color="#fffaf4"/>
    <stop offset="42%" stop-color="#ffe6d3"/>
    <stop offset="74%" stop-color="#f0ddd1"/>
    <stop offset="100%" stop-color="#e6e3de"/>
  </radialGradient>
  <radialGradient id="glow" cx="50%" cy="60%" r="46%">
    <stop offset="0%" stop-color="#ff9e63" stop-opacity=".9"/>
    <stop offset="55%" stop-color="#ffb37e" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="rim" cx="50%" cy="50%" r="50%">
    <stop offset="74%" stop-color="#ffffff" stop-opacity="0"/>
    <stop offset="93%" stop-color="#ffffff" stop-opacity=".85"/>
    <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
  </radialGradient>
  <filter id="speckle">
    <feTurbulence type="fractalNoise" baseFrequency="0.92" numOctaves="2" seed="6" stitchTiles="stitch"/>
    <feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 -1.35 0.9"/>
    <feGaussianBlur stdDeviation="0.25"/>
  </filter>
  <filter id="speckle2">
    <feTurbulence type="fractalNoise" baseFrequency="0.45" numOctaves="2" seed="11" stitchTiles="stitch"/>
    <feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 -1.5 0.95"/>
  </filter>
  <filter id="soft"><feGaussianBlur stdDeviation="9"/></filter>
  <clipPath id="ball"><circle cx="300" cy="296" r="186"/></clipPath>
 </defs>
 <ellipse cx="300" cy="536" rx="158" ry="20" fill="#7a756e" opacity=".16" filter="url(#soft)"/>
 <circle cx="300" cy="296" r="186" fill="url(#sphere)"/>
 <circle cx="300" cy="296" r="186" fill="url(#glow)"/>
 <rect x="100" y="96" width="400" height="400" fill="#ffffff" opacity=".62"
       filter="url(#speckle)" clip-path="url(#ball)"/>
 <rect x="100" y="96" width="400" height="400" fill="#ffffff" opacity=".5"
       filter="url(#speckle2)" clip-path="url(#ball)"/>
 <circle cx="300" cy="296" r="186" fill="url(#rim)"/>
</svg>
"""

HERO = """
<div class="topbar">
  <div class="brand">M<span class="dot"></span>X</div>
</div>

<section class="hero">
  <div class="orb-wrap">__ORB__</div>
  <div class="hero-grid">
    <div class="hero-left">
      <div class="eyebrow">Concrete Strength Studio · <b>Gradient Boosting</b></div>
      <h1 class="h-title">
        <span class="l1">재료의 배합,</span>
        <span class="l2">강도의 <em>예측.</em></span>
      </h1>
      <p class="h-sub">시멘트, 감수제, 재령, 물 — 네 가지 배합 변수만으로 콘크리트의
      압축강도를 예측하는 머신러닝 스튜디오. 숫자가 강도가 되는 순간을 설계합니다.</p>
      <a class="cta" href="#studio">강도 예측하기 <span class="ar">&#8599;</span></a>
    </div>
    <div class="hero-right">
      <p class="hr-copy"><b>시멘트량</b>, <b>고성능 감수제량</b>, <b>재령 기간</b>, 그리고
      <b>물양</b> &mdash; 이 네 값의 균형이 콘크리트가 견디는 힘을 결정합니다.</p>
      <div class="tags">
        <span class="tag">시멘트</span>
        <span class="tag">감수제</span>
        <span class="tag">재령</span>
        <span class="tag">물</span>
        <span class="tag plus">MPa +</span>
      </div>
    </div>
  </div>
</section>
""".replace("__ORB__", ORB_SVG)
st.markdown(HERO, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
#  STUDIO  (입력 + 예측)
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div id="studio" class="sec">
  <div>
    <div class="sec-k">The Mix Studio</div>
    <h2 class="sec-t">배합비를 조절해 <em>강도를 예측</em>하세요.</h2>
  </div>
  <p class="sec-d">값을 직접 입력(또는 +/− 버튼)하면 결과가 즉시 갱신됩니다. 모든 값은 Kaggle
  콘크리트 압축강도 데이터셋의 실제 분포 범위 안에서 조정됩니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 0.88], gap="large")


def input_row(idx, kr, en, unit, lo, hi, default, step, fmt="%.0f", key=None):
    """라벨(상단) + 키보드로 직접 입력 가능한 숫자 입력칸(하단)."""
    rng = f"{fmt % lo}–{fmt % hi} {unit}"
    st.markdown(
        f"""<div class="ic">
              <div class="ic-l"><span class="ic-kr">{idx} · {kr}</span>
              <span class="ic-en">{en} · {unit}</span></div>
              <div class="ic-rng">{rng}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    val = st.number_input(
        kr, min_value=lo, max_value=hi, value=default, step=step,
        format=fmt, key=key, label_visibility="collapsed",
    )
    return val


with left:
    st.markdown(
        """<div class="panel-h"><span class="idx">01—04</span>
        <span class="t">배합 변수 입력</span>
        <span class="s">Mix proportions</span></div>""",
        unsafe_allow_html=True,
    )
    cement = input_row("01", "시멘트량", "Cement", "kg/m³", 100.0, 600.0, 281.0, 1.0, fmt="%.0f", key="c")
    superp = input_row("02", "고성능 감수제량", "Superplasticizer", "kg/m³", 0.0, 35.0, 6.0, 0.1, fmt="%.1f", key="s")
    age = input_row("03", "재령 기간", "Age", "days", 1, 200, 28, 1, fmt="%d", key="a")
    water = input_row("04", "물양", "Water", "kg/m³", 120.0, 250.0, 182.0, 0.5, fmt="%.1f", key="w")


# ── 예측 (학습 시 컬럼 순서 그대로) ──
X = pd.DataFrame([[cement, superp, age, water]], columns=FEATURES)
strength = float(model.predict(X)[0])
strength = max(0.0, strength)


def grade(mpa):
    if mpa < 20:
        return "저강도", "비구조용 · 채움/보조재 수준"
    if mpa < 30:
        return "보통", "주택 기초 등 일반 구조용"
    if mpa < 40:
        return "표준", "일반 건축 구조용 강도"
    if mpa < 50:
        return "고강도", "고층 · 교량 등 구조용"
    return "초고강도", "특수 구조 · 프리캐스트"


g_label, g_desc = grade(strength)
GMAX = 80.0
pos = max(2.0, min(98.0, strength / GMAX * 100.0))
wc_ratio = water / cement if cement else 0.0  # 물–시멘트비 (낮을수록 강함)

with right:
    st.markdown(
        f"""
        <div class="res-lab">예측 압축강도 · Predicted</div>
        <div class="res-big">{strength:.1f}<span class="u">MPa</span></div>
        <div class="res-grade"><span class="pip"></span>{g_label}</div>
        <div class="res-note">{g_desc}. 28일 기준 표준 콘크리트는 대략 24&ndash;40&nbsp;MPa 범위입니다.</div>

        <div class="gauge">
          <div class="gauge-bar"><div class="gauge-mk" style="left:{pos:.1f}%"></div></div>
          <div class="gauge-sc"><span>0</span><span>40</span><span>80&nbsp;MPa</span></div>
        </div>

        <div class="feat-mini">
          <div class="fm"><div class="k">물–시멘트비 W/C</div><div class="v">{wc_ratio:.2f}</div></div>
          <div class="fm"><div class="k">재령</div><div class="v">{int(age)}<span style="font-size:11px;font-weight:500"> 일</span></div></div>
          <div class="fm"><div class="k">감수제</div><div class="v">{superp:.1f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="foot">
  <div>
    <b>MIX — Concrete Strength Studio</b><br>
    Model · Gradient Boosting Regressor &nbsp;|&nbsp; Test R² ≈ 0.82<br>
    Active · {MODEL_SOURCE}
  </div>
  <div class="r">
    Data · Kaggle Concrete Compressive Strength<br>
    Type · 페이퍼로지 &amp; 서울알림체<br>
    Built with Streamlit &nbsp;&#8599;
  </div>
</div>
""",
    unsafe_allow_html=True,
)
