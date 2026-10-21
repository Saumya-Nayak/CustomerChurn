"""
Enterprise XAI Customer Churn Inspector
---------------------------------------
Run:      streamlit run app.py
Install:  pip install streamlit pandas numpy scikit-learn plotly shap anthropic
Secret:   .streamlit/secrets.toml  ->  ANTHROPIC_API_KEY = "sk-ant-..."
          (or paste the key into the sidebar field at runtime)
"""

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# --------------------------------------------------------------------------- #
# 0. Theme: force a light theme before anything renders
# --------------------------------------------------------------------------- #
PRIMARY, PRIMARY_DARK = "#2563EB", "#1D4ED8"
for _key, _val in {
    "theme.base": "light",
    "theme.primaryColor": PRIMARY,
    "theme.backgroundColor": "#FFFFFF",
    "theme.secondaryBackgroundColor": "#F8FAFC",
    "theme.textColor": "#0F172A",
    "theme.font": "sans serif",
}.items():
    try:
        st._config.set_option(_key, _val)
    except Exception:
        pass  # CSS below still enforces the light look

_cfg = Path(__file__).parent / ".streamlit" / "config.toml"
if not _cfg.exists():
    try:
        _cfg.parent.mkdir(exist_ok=True)
        _cfg.write_text('[theme]\nbase = "light"\nprimaryColor = "#2563EB"\nbackgroundColor = "#FFFFFF"\n'
                        'secondaryBackgroundColor = "#F8FAFC"\ntextColor = "#0F172A"\nfont = "sans serif"\n')
    except Exception:
        pass

st.set_page_config(
    page_title="Churn Inspector", page_icon="📉", layout="wide",
    initial_sidebar_state="expanded",
)

MODEL = "claude-sonnet-5-5"
HIGH, MEDIUM = 0.65, 0.35
TIERS = {"High": ("#B91C1C", "#FEE2E2"), "Medium": ("#B45309", "#FEF3C7"), "Low": ("#15803D", "#DCFCE7")}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ color-scheme: light; --primary:{PRIMARY}; }}
html, body, .stApp, .stApp :is(p, h1, h2, h3, h4, label, button, input, textarea, li, td, th, a, small),
[data-baseweb] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background:#FFFFFF !important; color:#0F172A !important; }}
[data-testid="stSidebar"] {{ background:#F8FAFC !important; border-right:1px solid #E2E8F0; }}
[data-testid="stSidebar"] *, .stApp p, .stApp label, .stApp li {{ color:#0F172A; }}
.block-container {{ padding-top:2rem; max-width:1280px; }}
h1, h2, h3 {{ letter-spacing:-0.02em; color:#0F172A !important; font-weight:650; }}

/* Primary action buttons */
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{
  background:{PRIMARY} !important; color:#FFFFFF !important; border:none !important;
  border-radius:8px !important; font-weight:600 !important; padding:0.6rem 1.1rem !important;
  box-shadow:0 1px 2px rgba(37,99,235,.25); transition:background .15s ease;
}}
.stButton > button:hover {{ background:{PRIMARY_DARK} !important; }}
.stButton > button p {{ color:#FFFFFF !important; }}
.stButton > button:focus-visible {{ outline:3px solid #BFDBFE !important; }}

/* Inputs */
[data-baseweb="select"] > div, [data-baseweb="input"], input[type="password"] {{
  background:#FFFFFF !important; border-color:#CBD5E1 !important; color:#0F172A !important; border-radius:8px !important;
}}
[data-baseweb="popover"] *, [data-baseweb="menu"] {{ background:#FFFFFF; color:#0F172A; }}
[data-testid="stFileUploaderDropzone"] {{ background:#FFFFFF !important; border:1.5px dashed #94A3B8 !important; border-radius:10px; }}
[data-testid="stFileUploaderDropzone"] button {{ background:#FFFFFF !important; color:{PRIMARY} !important; border:1px solid {PRIMARY} !important; box-shadow:none; }}

[data-baseweb="input"] > div, [data-baseweb="input"] button, [data-baseweb="base-input"] {{ background:#FFFFFF !important; color:#0F172A !important; }}
[data-baseweb="input"] svg {{ fill:#475569 !important; color:#475569 !important; }}
input {{ color:#0F172A !important; -webkit-text-fill-color:#0F172A !important; caret-color:#0F172A; }}
input::placeholder {{ color:#94A3B8 !important; -webkit-text-fill-color:#94A3B8 !important; }}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ border-radius:14px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#F8FAFC; border-color:#E2E8F0 !important; }}
.kpi {{ background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:16px 18px; height:100%; }}
.kpi .lbl {{ font-size:.8rem; color:#64748B; font-weight:500; }}
.kpi .val {{ font-size:1.75rem; font-weight:700; color:#0F172A; letter-spacing:-0.02em; line-height:1.25; }}
.kpi .sub {{ font-size:.78rem; color:#64748B; }}
.card-title {{ font-size:1.05rem; font-weight:650; color:#0F172A; margin:0 0 2px 0; }}
.card-sub {{ font-size:.85rem; color:#64748B; margin:0 0 10px 0; }}
.tile {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:10px 12px; }}
.tile .lbl {{ font-size:.75rem; color:#64748B; }}
.tile .val {{ font-size:1.05rem; font-weight:600; color:#0F172A; }}
.tile .delta {{ font-size:.74rem; color:#64748B; }}
.badge {{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:.82rem; }}
.alert-xai {{ background:#EFF6FF; border:1px solid #BFDBFE; border-left:6px solid {PRIMARY}; border-radius:10px; padding:18px 20px; }}
.alert-xai .hd {{ font-weight:700; color:#1E3A8A; margin-bottom:6px; font-size:.95rem; }}
.alert-xai .bd {{ color:#0F172A; font-size:1.02rem; line-height:1.6; }}
.alert-err {{ background:#FEF2F2; border:1px solid #FECACA; border-left:6px solid #DC2626; border-radius:10px; padding:14px 18px; color:#7F1D1D; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def make_synthetic(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    bill = np.round(rng.normal(70, 22, n).clip(20, 140), 2)
    calls = rng.poisson(2.2, n).clip(0, 9)
    usage = np.round(rng.gamma(4, 55, n).clip(5, 900), 1)
    logit = (-2.1 + 1.5 * (contract == "Month-to-month") - 1.1 * (contract == "Two year")
             + 0.45 * calls + 0.018 * (bill - 70) - 0.002 * (usage - 220))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "Customer_ID": [f"CUST-{1001 + i}" for i in range(n)],
        "Monthly_Bill": bill, "Contract_Type": contract,
        "Customer_Service_Calls": calls, "Total_Data_Usage_GB": usage, "Churn": churn,
    })


def prepare(df: pd.DataFrame):
    """Detect ID / target columns and return a cleaned frame."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target = next((c for c in df.columns if c.lower() == "churn"), None)
    if target is None:
        raise ValueError("The CSV needs a 'Churn' column (0/1 or Yes/No).")
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.lower().isin({"1", "yes", "y", "true", "churned"})
    df[target] = y.astype(int)
    if df[target].nunique() < 2:
        raise ValueError("'Churn' must contain both 0 and 1 values to train a model.")

    id_col = next((c for c in df.columns if c.lower().replace(" ", "_") in ("customer_id", "customerid", "id")), None)
    if id_col is None:
        id_col = next((c for c in df.columns if "id" in c.lower() and c != target), None)
    if id_col is None:
        id_col = "Customer_ID"
        df.insert(0, id_col, [f"ROW-{i + 1}" for i in range(len(df))])
    df[id_col] = df[id_col].astype(str)
    df = df.drop_duplicates(subset=id_col).reset_index(drop=True)
    return df, id_col, target


@st.cache_resource(show_spinner="Training Random Forest…")
def train(df: pd.DataFrame, id_col: str, target: str):
    X = df.drop(columns=[id_col, target])
    # Drop free-text style columns (almost all unique)
    X = X.drop(columns=[c for c in X.columns
                        if X[c].dtype == object and X[c].nunique() > max(50, 0.5 * len(X))])
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    X[cat] = X[cat].astype(str)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat),
    ])
    make = lambda: Pipeline([("pre", pre), ("rf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1))])

    y = df[target]
    auc = None
    if y.value_counts().min() >= 2 and len(df) >= 40:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        auc = roc_auc_score(yte, make().fit(Xtr, ytr).predict_proba(Xte)[:, 1])

    model = make().fit(X, y)
    names = num + cat  # same order the ColumnTransformer emits
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names).sort_values()
    risk = pd.Series(model.predict_proba(X)[:, 1], index=df[id_col])

    # Local explainability: SHAP values (in churn-probability units) per customer
    shap_df = None
    try:
        import shap
        sv = shap.TreeExplainer(model.named_steps["rf"]).shap_values(model.named_steps["pre"].transform(X))
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        if sv.ndim == 3:  # newer shap: (rows, features, classes)
            sv = sv[:, :, 1]
        shap_df = pd.DataFrame(sv, index=df[id_col], columns=names)
    except Exception:
        shap_df = None  # app falls back to non-SHAP explanations
    return model, imp, risk, auc, num, cat, shap_df


# --------------------------------------------------------------------------- #
# 2. Helpers
# --------------------------------------------------------------------------- #
def tier_of(p: float) -> str:
    return "High" if p >= HIGH else "Medium" if p >= MEDIUM else "Low"


def fmt(v) -> str:
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v):,}"
    return f"{v:,.2f}" if isinstance(v, (float, np.floating)) else html.escape(str(v))


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def alert(text: str, title: str = "Claude risk explanation") -> str:
    body = html.escape(text).replace("\n", "<br>")
    return f'<div class="alert-xai"><div class="hd">{title}</div><div class="bd">{body}</div></div>'


def local_explanation(row: pd.Series, p: float, df: pd.DataFrame, num: list, imp: pd.Series, contrib=None) -> str:
    """Free, offline 3-sentence explanation used when no API key is available."""
    signals = []
    if contrib is not None:
        for f, c in contrib[contrib > 0.01].sort_values(ascending=False).head(3).items():
            signals.append(f"{f.replace('_', ' ').lower()} ({row[f]}) adds {c * 100:.0f} points to the risk")
    for f in ([] if signals else imp.sort_values(ascending=False).index):
        v, name = row[f], f.replace("_", " ").lower()
        if f in num:
            if pd.notna(v) and df[f].mean() and (float(v) / float(df[f].mean()) - 1) >= 0.20:
                signals.append(f"{name} is {(float(v) / float(df[f].mean()) - 1) * 100:.0f}% above the portfolio average")
        else:
            rate = df.loc[df[f].astype(str) == str(v), TARGET].mean()
            if rate > df[TARGET].mean() * 1.2:
                signals.append(f"{name} is '{v}', a segment where {rate:.0%} of customers churn")
    why = "; ".join(signals[:3]) or "no single metric stands out against the portfolio average"
    step = {"High": "Have an account manager call this customer within a week with a tailored retention offer or contract upgrade.",
            "Medium": "Send a proactive check-in and a light incentive before the account slips further.",
            "Low": "Keep the account in routine engagement and review it again next quarter."}[tier_of(p)]
    return f"This customer has a {p:.0%} predicted churn risk ({tier_of(p)}). The main warning signs: {why}. {step}"


def get_api_key(fallback: str):
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return fallback or None


def build_prompt(row: pd.Series, p: float, df: pd.DataFrame, feats: list, drivers: list, contrib=None) -> str:
    lines = []
    for f in feats:
        v = row[f]
        if pd.api.types.is_numeric_dtype(df[f]):
            lines.append(f"- {f}: {fmt(v)} (portfolio average {fmt(float(df[f].mean()))})")
        else:
            lines.append(f"- {f}: {v}")
    shap_txt = ""
    if contrib is not None:
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
        shap_txt = ("SHAP contributions to this customer's churn probability (percentage points; positive raises risk): "
                    + ", ".join(f"{f}={row[f]}: {c * 100:+.1f}" for f, c in top.items()) + ".\n\n")
    return (
        f"Customer {row.name} has a model-predicted churn probability of {p:.0%} ({tier_of(p)} risk).\n"
        "Customer metrics:\n" + "\n".join(lines) + "\n\n"
        f"Top global churn drivers in our model: {', '.join(drivers)}.\n\n" + shap_txt +
        "Write exactly three sentences of plain English for a non-technical account manager: "
        "sentences 1-2 explain why this customer is at risk, grounded in the SHAP contributions when given and referencing their specific metrics; "
        "sentence 3 gives one concrete retention next step. No bullets, headings, or preamble."
    )


# --------------------------------------------------------------------------- #
# 3. Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Data source")
    upload = st.file_uploader("Customer CSV", type=["csv"], help="Must include a 'Churn' column (0/1 or Yes/No).")
    st.markdown("### Claude access")
    secret_key = get_api_key("")
    if secret_key:
        st.success("API key loaded from Streamlit secrets.")
        sidebar_key = ""
    else:
        sidebar_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-…")

# --------------------------------------------------------------------------- #
# 4. Load + train
# --------------------------------------------------------------------------- #
try:
    raw = pd.read_csv(upload) if upload else make_synthetic()
    df, ID, TARGET = prepare(raw)
    model, importances, risk, auc, num_cols, cat_cols, shap_df = train(df, ID, TARGET)
except Exception as exc:
    st.markdown(f'<div class="alert-err"><b>Could not process this file.</b><br>{html.escape(str(exc))}</div>',
                unsafe_allow_html=True)
    st.stop()

feats = num_cols + cat_cols
high_risk = int((risk >= HIGH).sum())
bill_col = next((c for c in num_cols if "bill" in c.lower() or "charge" in c.lower()), None)

# --------------------------------------------------------------------------- #
# 5. Dashboard
# --------------------------------------------------------------------------- #
st.markdown("# Customer Churn Inspector")
st.markdown('<p class="card-sub">Explainable churn risk across your customer base, with plain-English audits for each account.'
            + ("" if upload else " Showing a synthetic sample of 200 customers; upload a CSV to use your own data.")
            + "</p>", unsafe_allow_html=True)

cols = st.columns(5)
tiles = [
    ("Customers", f"{len(df):,}", "In current dataset"),
    ("Observed churn", f"{df[TARGET].mean():.1%}", f"{int(df[TARGET].sum()):,} churned accounts"),
    ("High-risk accounts", f"{high_risk:,}", f"Predicted risk ≥ {HIGH:.0%}"),
    ("Monthly revenue at risk" if bill_col else "Average risk score",
     f"{df.loc[df[ID].isin(risk[risk >= HIGH].index), bill_col].sum():,.0f}" if bill_col else f"{risk.mean():.1%}",
     f"Sum of {bill_col} for high-risk" if bill_col else "Mean predicted probability"),
    ("Model ROC AUC", f"{auc:.2f}" if auc is not None else "n/a", "25% hold-out" if auc is not None else "Too little data to validate"),
]
for c, t in zip(cols, tiles):
    c.markdown(kpi(*t), unsafe_allow_html=True)

st.write("")
with st.container(border=True):
    st.markdown('<div class="card-title">Global risk drivers</div>'
                '<div class="card-sub">Share of the Random Forest’s predictive signal contributed by each feature.</div>',
                unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=importances.values, y=[n.replace("_", " ") for n in importances.index], orientation="h",
        marker=dict(color=PRIMARY), text=[f"{v:.1%}" for v in importances.values], textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>", cliponaxis=False,
    ))
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", color="#0F172A", size=13),
        height=max(260, 54 * len(importances) + 60), margin=dict(l=8, r=40, t=8, b=8),
        xaxis=dict(tickformat=".0%", gridcolor="#E2E8F0", zeroline=False, range=[0, importances.max() * 1.2]),
        yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A")),
    )
    st.plotly_chart(fig, theme=None, use_container_width=True, config={"displayModeBar": False})

st.write("")
st.markdown("## Customer audit")
with st.container(border=True):
    order = risk.sort_values(ascending=False).index.tolist()
    cid = st.selectbox("Search or select a customer ID", order,
                       format_func=lambda i: f"{i}  ·  {risk[i]:.0%} risk",
                       help="Customers are sorted from highest to lowest predicted risk.")
    row = df.set_index(ID).loc[cid]
    contrib = shap_df.loc[cid] if shap_df is not None else None
    p = float(risk[cid])
    tier = tier_of(p)
    fg, bg = TIERS[tier]

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown(
            f'<div class="card-title">{html.escape(str(cid))} '
            f'<span class="badge" style="color:{fg};background:{bg}">{tier} risk</span></div>'
            f'<div class="card-sub">Predicted churn probability <b>{p:.1%}</b> · '
            f'Observed outcome: <b>{"Churned" if row[TARGET] == 1 else "Active"}</b></div>',
            unsafe_allow_html=True)
        grid = st.columns(2)
        for i, f in enumerate(feats[:12]):
            v = row[f]
            delta = ""
            if f in num_cols and pd.notna(v) and df[f].mean():
                d = (float(v) / float(df[f].mean()) - 1) * 100
                delta = f'<div class="delta">{d:+.0f}% vs portfolio avg</div>'
            grid[i % 2].markdown(
                f'<div class="tile"><div class="lbl">{html.escape(f.replace("_", " "))}</div>'
                f'<div class="val">{fmt(v) if pd.notna(v) else "—"}</div>{delta}</div>',
                unsafe_allow_html=True)
            grid[i % 2].write("")

    with right:
        st.markdown('<div class="card-title">AI risk explanation</div>'
                    '<div class="card-sub">Three sentences: why this account is at risk and what to do next.</div>',
                    unsafe_allow_html=True)
        store = st.session_state.setdefault("explanations", {})
        clicked = st.button("Generate Claude Risk Explanation", type="primary", use_container_width=True)
        slot = st.empty()

        if clicked:
            key = get_api_key(sidebar_key)
            if not key:
                slot.markdown(alert(local_explanation(row, p, df, num_cols, importances, contrib),
                                    "Rule-based explanation (no API key set)"), unsafe_allow_html=True)
            else:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=key)
                    drivers = [n.replace("_", " ") for n in importances.sort_values(ascending=False).index[:3]]
                    prompt = build_prompt(row, p, df, feats, drivers, contrib)
                    text = ""
                    slot.markdown(alert("Analyzing account…"), unsafe_allow_html=True)
                    with client.messages.stream(
                        model=MODEL, max_tokens=350,
                        system="You are a customer retention analyst. Be specific, plain-spoken, and concise.",
                        messages=[{"role": "user", "content": prompt}],
                    ) as stream:
                        for chunk in stream.text_stream:
                            text += chunk
                            slot.markdown(alert(text), unsafe_allow_html=True)
                    store[(cid, p)] = text
                except ImportError:
                    slot.markdown('<div class="alert-err"><b>Missing package.</b> Run <code>pip install anthropic</code>.</div>',
                                  unsafe_allow_html=True)
                except Exception as exc:  # auth, rate limit, network, etc.
                    slot.markdown(f'<div class="alert-err"><b>Claude request failed.</b><br>{html.escape(str(exc))}</div>',
                                  unsafe_allow_html=True)
        elif (cid, p) in store:
            slot.markdown(alert(store[(cid, p)]), unsafe_allow_html=True)

    if contrib is not None:
        st.markdown("---")
        st.markdown('<div class="card-title">Why this score (SHAP)</div>'
                    f'<div class="card-sub">Each bar is how many percentage points a feature pushed this customer’s churn risk '
                    f'up (red) or down (green). Starting from the portfolio average of {risk.mean():.0%}, the bars roughly add up to '
                    f'this customer’s {p:.0%}.</div>', unsafe_allow_html=True)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        labels = [f"{f.replace('_', ' ')} = {fmt(row[f]) if pd.notna(row[f]) else '—'}" for f in top.index]
        pts = top.values * 100
        sfig = go.Figure(go.Bar(
            x=pts, y=labels, orientation="h",
            marker=dict(color=["#DC2626" if v > 0 else "#16A34A" for v in pts]),
            text=[f"{v:+.1f}" for v in pts], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
        span = max(abs(pts).max(), 1) * 1.3
        sfig.update_layout(
            template="plotly_white", paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
            font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", color="#0F172A", size=13),
            height=max(240, 46 * len(top) + 70), margin=dict(l=8, r=40, t=8, b=8),
            xaxis=dict(title="Impact on churn probability (percentage points)", gridcolor="#E2E8F0",
                       zerolinecolor="#94A3B8", range=[-span, span]),
            yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A")))
        st.plotly_chart(sfig, theme=None, use_container_width=True, config={"displayModeBar": False})
