"""
Churn Prediction Dashboard (ANN) — pure Python (Streamlit + Plotly), no HTML/CSS/JS.

Run with:  streamlit run app.py
"""

import pickle

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence | ANN Dashboard",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded",
)

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}

# ---------------------------------------------------------------------------
# THEMES — pure-Python palettes (Streamlit's native color-markdown + Plotly
# colors). No CSS/HTML involved; switching is instant, no page reload.
# ---------------------------------------------------------------------------
THEMES = {
    "Ocean Blue": {
        "accent": "blue",
        "icon": "🌊",
        "gauge_bar": "#2563EB",
        "gauge_zones": ["rgba(34,197,94,0.15)", "rgba(245,158,11,0.15)", "rgba(239,68,68,0.15)"],
        "tab_icons": ["🧑", "💳", "🔁"],
    },
    "Sunset": {
        "accent": "orange",
        "icon": "🌅",
        "gauge_bar": "#F97316",
        "gauge_zones": ["rgba(250,204,21,0.18)", "rgba(251,146,60,0.18)", "rgba(220,38,38,0.18)"],
        "tab_icons": ["🧑", "💳", "🔁"],
    },
    "Emerald": {
        "accent": "green",
        "icon": "🌿",
        "gauge_bar": "#10B981",
        "gauge_zones": ["rgba(16,185,129,0.15)", "rgba(234,179,8,0.15)", "rgba(239,68,68,0.15)"],
        "tab_icons": ["🧑", "💳", "🔁"],
    },
    "Royal Violet": {
        "accent": "violet",
        "icon": "🔮",
        "gauge_bar": "#8B5CF6",
        "gauge_zones": ["rgba(34,197,94,0.15)", "rgba(139,92,246,0.15)", "rgba(239,68,68,0.15)"],
        "tab_icons": ["🧑", "💳", "🔁"],
    },
    "Rose": {
        "accent": "red",
        "icon": "🌹",
        "gauge_bar": "#EC4899",
        "gauge_zones": ["rgba(34,197,94,0.15)", "rgba(236,72,153,0.15)", "rgba(220,38,38,0.15)"],
        "tab_icons": ["🧑", "💳", "🔁"],
    },
}


# ---------------------------------------------------------------------------
# LOAD ARTIFACTS (cached so they only load once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_ann_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


model, scaler, encoders = load_artifacts()
le_gender = encoders["le_gender"]
ohe_geo = encoders["ohe_geo"]
FEATURE_COLUMNS = encoders["feature_columns"]
GEO_CATEGORIES = encoders["geo_categories"]


def preprocess_single(record: dict) -> np.ndarray:
    row = {
        "CreditScore": record["CreditScore"],
        "Gender": le_gender.transform([record["Gender"]])[0],
        "Age": record["Age"],
        "Tenure": record["Tenure"],
        "Balance": record["Balance"],
        "NumOfProducts": record["NumOfProducts"],
        "HasCrCard": record["HasCrCard"],
        "IsActiveMember": record["IsActiveMember"],
        "EstimatedSalary": record["EstimatedSalary"],
    }
    geo_encoded = ohe_geo.transform([[record["Geography"]]])[0]
    for i, cat in enumerate(GEO_CATEGORIES[1:]):
        row[f"Geography_{cat}"] = geo_encoded[i]

    ordered = np.array([[row[c] for c in FEATURE_COLUMNS]])
    return scaler.transform(ordered)


def risk_tier(prob: float) -> tuple[str, str, str]:
    """Return (label, semantic_color_hex, streamlit_color_name) for a churn probability."""
    if prob < RISK_THRESHOLDS["low"]:
        return "Low Risk", "#22C55E", "green"
    elif prob < RISK_THRESHOLDS["medium"]:
        return "Medium Risk", "#F59E0B", "orange"
    return "High Risk", "#EF4444", "red"


def make_gauge(prob: float, bar_color: str, zones: list, needle_color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#E6EDF3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B949E", "tickfont": {"color": "#8B949E"}},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#161B22",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": zones[0]},
                    {"range": [30, 60], "color": zones[1]},
                    {"range": [60, 100], "color": zones[2]},
                ],
                "threshold": {
                    "line": {"color": needle_color, "width": 3},
                    "thickness": 0.9,
                    "value": prob * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6EDF3"},
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR — branding, theme switcher & context
# ---------------------------------------------------------------------------
with st.sidebar:
    theme_name = st.selectbox("🎨 Dashboard Theme", list(THEMES.keys()), index=0)
    theme = THEMES[theme_name]

    st.markdown(f"### {theme['icon']} Churn Intelligence")
    st.caption("ANN-powered customer retention tool")
    st.divider()

    st.markdown(f":{theme['accent']}[**How it works**]")
    st.write(
        "This tool scores a customer's likelihood of churning using an "
        "Artificial Neural Network trained on historical banking customer data."
    )

    st.markdown(f":{theme['accent']}[**Risk tiers**]")
    st.markdown("🟢 **Low** — under 30%")
    st.markdown("🟠 **Medium** — 30–60%")
    st.markdown("🔴 **High** — above 60%")
    st.divider()
    st.caption("Model: Dense ANN (Keras/TensorFlow) · Trained on 10,000 customer records")


# ---------------------------------------------------------------------------
# MAIN — header
# ---------------------------------------------------------------------------
st.markdown(f"## {theme['icon']} :{theme['accent']}[Customer Churn Prediction]")
st.caption("Enter a customer's profile below to estimate their probability of churning.")
st.divider()

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(f"#### :{theme['accent']}[Customer Profile]")

    icons = theme["tab_icons"]
    tab1, tab2, tab3 = st.tabs([f"{icons[0]} Demographics", f"{icons[1]} Account Details", f"{icons[2]} Engagement"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 40)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            geography = st.selectbox("Geography", GEO_CATEGORIES)
            credit_score = st.slider("Credit Score", 300, 900, 650)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0, step=1000.0)
            salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0, step=1000.0)
        with col2:
            tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
            num_products = st.slider("Number of Products", 1, 4, 1)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        with col2:
            is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

    st.write("")
    predict_clicked = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
if predict_clicked:
    record = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }
    X = preprocess_single(record)
    prob = float(model.predict(X, verbose=0)[0][0])
    label, hex_color, st_color = risk_tier(prob)

    st.write("")
    with st.container(border=True):
        st.markdown(f"#### :{theme['accent']}[Prediction Result]")
        c1, c2 = st.columns([1, 1])

        with c1:
            fig = make_gauge(prob, theme["gauge_bar"], theme["gauge_zones"], hex_color)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            emoji = "🟢" if label == "Low Risk" else "🟠" if label == "Medium Risk" else "🔴"
            st.markdown(f"##### {emoji} :{st_color}[{label}]")
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")

            if label == "High Risk":
                st.error("Recommend proactive retention outreach.")
            elif label == "Medium Risk":
                st.warning("Monitor closely; consider a check-in.")
            else:
                st.success("Customer appears stable.")

        st.caption(f"Model decision threshold: 50% · Raw probability: {prob:.4f}")
