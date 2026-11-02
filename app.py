import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# FRAUDGUARD | ADVANCED AI FINANCIAL RISK MONITORING
# Built around the existing Random Forest model and artifacts.
# ============================================================

st.set_page_config(
    page_title="FraudGuard | AI Risk Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "random_forest_model.pkl"
THRESHOLD_PATH = APP_DIR / "fraud_threshold.pkl"
FEATURE_PATH = APP_DIR / "feature_names.pkl"
DATA_PATH = APP_DIR / "dashboard_transactions.csv"

RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": "#16A34A",
    "Medium": "#EAB308",
    "High": "#F97316",
    "Critical": "#DC2626",
}


# ============================================================
# 1. PROFESSIONAL FINTECH UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F4F6FB;
        color: #1E293B;
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101C36 0%, #172B4D 100%);
        border-right: 1px solid #263B5C;
    }
    [data-testid="stSidebar"] * { color: #F8FAFC; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #CBD5E1;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 7px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 10px;
        padding: 10px 12px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(59,130,246,.20);
    }
    h1 {
        color: #14213D !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
        letter-spacing: -.6px;
    }
    h2, h3 { color: #172B4D !important; }
    [data-testid="stCaptionContainer"] {
        color: #64748B;
        font-size: .9rem;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 15px;
        padding: 18px 19px;
        min-height: 118px;
        box-shadow: 0 4px 14px rgba(15,23,42,.045);
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: .83rem !important;
        font-weight: 650 !important;
    }
    [data-testid="stMetricValue"] {
        color: #14213D !important;
        font-size: 1.65rem !important;
        font-weight: 750 !important;
    }
    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 15px;
        padding: 8px;
        box-shadow: 0 4px 14px rgba(15,23,42,.035);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stTextInput"] > div > div {
        background: #FFFFFF;
        border-radius: 9px;
    }
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #93A4BC;
        border-radius: 13px;
        padding: 10px;
    }
    .stButton > button, .stDownloadButton > button {
        background: #1D4ED8;
        color: #FFFFFF;
        border: 1px solid #1D4ED8;
        border-radius: 9px;
        font-weight: 650;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #1E40AF;
        border-color: #1E40AF;
        color: #FFFFFF;
    }
    [data-testid="stAlert"], [data-testid="stExpander"] {
        border-radius: 11px;
    }
    hr { border-color: #E2E8F0 !important; }
    footer { visibility: hidden; }
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.55rem !important; }
        [data-testid="stMetric"] { padding: 13px; min-height: 100px; }
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. HELPERS
# ============================================================

def risk_category(score):
    """Map an unrounded 0-100 model score to a display risk band."""
    if score < 25:
        return "Low"
    if score < 60:
        return "Medium"
    if score < 80:
        return "High"
    return "Critical"


def model_scores(fitted_model, features_frame):
    """Return the probability/score for the positive fraud class."""
    probabilities = fitted_model.predict_proba(features_frame)
    classes = list(getattr(fitted_model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("The loaded model does not contain positive class 1.")
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def style_plot(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(color="#24324B"),
        legend_title_text="",
    )
    if height:
        fig.update_layout(height=height)
    return fig


def page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.divider()


def download_csv(frame, label, filename, key, include_index=False):
    st.download_button(
        label=label,
        data=frame.to_csv(index=include_index).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def safe_numeric_frame(frame, columns):
    converted = frame[columns].copy()
    for col in columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    if converted.isna().any().any():
        bad = converted.columns[converted.isna().any()].tolist()
        raise ValueError("Missing or non-numeric values in: " + ", ".join(bad))
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite or invalid numeric values.")
    return converted


# ============================================================
# 3. LOAD AND VALIDATE MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)


@st.cache_data
def load_metadata(threshold_path, feature_path, threshold_mtime, feature_mtime):
    return joblib.load(threshold_path), joblib.load(feature_path)


@st.cache_data
def load_dashboard_data(data_path, modified_time):
    return pd.read_csv(data_path, index_col=0)


required_paths = [MODEL_PATH, THRESHOLD_PATH, FEATURE_PATH, DATA_PATH]
missing_paths = [p.name for p in required_paths if not p.exists()]
if missing_paths:
    st.error("Required project files are missing: " + ", ".join(missing_paths))
    st.info("Keep app.py, the model, threshold, feature list and dashboard CSV together.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    threshold, feature_names = load_metadata(
        str(THRESHOLD_PATH),
        str(FEATURE_PATH),
        THRESHOLD_PATH.stat().st_mtime,
        FEATURE_PATH.stat().st_mtime,
    )
    df = load_dashboard_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load FraudGuard project files: {exc}")
    st.stop()

if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
    st.error("Saved decision threshold is not numeric.")
    st.stop()
if not np.isfinite(threshold) or not 0 < float(threshold) < 1:
    st.error("Saved decision threshold must be between 0 and 1.")
    st.stop()
threshold = float(threshold)

if not isinstance(feature_names, (list, tuple)) or not feature_names:
    st.error("Saved feature list is invalid.")
    st.stop()
feature_names = list(feature_names)

if not all(isinstance(x, str) and x.strip() for x in feature_names):
    st.error("Feature names must be non-empty strings.")
    st.stop()
if len(feature_names) != len(set(feature_names)):
    st.error("Feature list contains duplicate feature names.")
    st.stop()
if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_names):
    st.error("Model input feature count does not match the saved feature list.")
    st.stop()
if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support predict_proba().")
    st.stop()
if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != feature_names:
    st.error("Model feature names/order do not match feature_names.pkl.")
    st.stop()

required_columns = [
    "Time", "Amount", "Fraud_Probability",
    "Risk_Score", "Predicted_Fraud", "Risk_Category",
]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error("Dashboard dataset is missing columns: " + ", ".join(missing_columns))
    st.stop()
if df.empty:
    st.error("Dashboard dataset is empty.")
    st.stop()

numeric_columns = [
    "Time", "Amount", "Fraud_Probability", "Risk_Score", "Predicted_Fraud"
]
try:
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_columns].isna().any().any():
        raise ValueError("Dashboard data contains missing/non-numeric values.")
    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Dashboard data contains infinite values.")
except Exception as exc:
    st.error(str(exc))
    st.stop()

if (df["Amount"] < 0).any():
    st.error("Dashboard contains negative transaction amounts.")
    st.stop()
if not df["Fraud_Probability"].between(0, 1).all():
    st.error("Fraud_Probability values must be between 0 and 1.")
    st.stop()
if not df["Risk_Score"].between(0, 100).all():
    st.error("Risk_Score values must be between 0 and 100.")
    st.stop()
if not df["Predicted_Fraud"].isin([0, 1]).all():
    st.error("Predicted_Fraud must contain only 0 and 1.")
    st.stop()
if not df["Risk_Category"].isin(RISK_ORDER).all():
    st.error("Dashboard contains an unknown risk category.")
    st.stop()

df["Predicted_Fraud"] = df["Predicted_Fraud"].astype(int)
expected_categories = df["Risk_Score"].apply(risk_category)
if not (expected_categories.to_numpy() == df["Risk_Category"].astype(str).to_numpy()).all():
    st.warning("Some saved risk categories do not match the displayed risk scores.")

expected_predictions = (df["Fraud_Probability"] >= threshold).astype(int)
if not np.array_equal(expected_predictions.to_numpy(), df["Predicted_Fraud"].to_numpy()):
    st.warning("Some saved model decisions differ from the configured threshold.")

# Keep the original order for category charts.
df["Risk_Category"] = pd.Categorical(df["Risk_Category"], categories=RISK_ORDER, ordered=True)


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 18px 0;">
          <div style="font-size:27px;font-weight:800;color:#fff;letter-spacing:-1px;">
            🛡️ FraudGuard
          </div>
          <div style="font-size:11px;color:#AFC2DD;letter-spacing:1.5px;margin-top:5px;">
            AI FINANCIAL RISK MONITORING
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**WORKSPACE**")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Transaction Monitoring",
            "Risk Analytics",
            "Explainable AI",
            "Reports & Alerts",
            "Upload & Predict",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**MODEL INFORMATION**")
    st.write("Random Forest Classifier")
    st.write(f"Decision threshold: **{threshold * 100:.0f}/100**")
    st.caption(f"{len(feature_names)} input features")
    st.divider()
    st.caption("Academic ML Project | 2026")


# ============================================================
# 5. SHARED METRICS
# ============================================================

total_transactions = len(df)
flagged_count = int(df["Predicted_Fraud"].sum())
normal_count = total_transactions - flagged_count
critical_count = int((df["Risk_Category"] == "Critical").sum())
high_count = int((df["Risk_Category"] == "High").sum())
high_critical_count = high_count + critical_count
average_risk = float(df["Risk_Score"].mean())
flagged_amount = float(df.loc[df["Predicted_Fraud"] == 1, "Amount"].sum())


# ============================================================
# 6. OVERVIEW
# ============================================================

if page == "Overview":
    st.title("Financial Risk Overview")
    st.caption("Executive dashboard | AI-powered transaction monitoring and risk assessment")
    st.divider()

    st.subheader("Key Performance Indicators")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total_transactions:,}")
    c2.metric("Flagged Predictions", f"{flagged_count:,}", f"{flagged_count / total_transactions * 100:.2f}% of transactions")
    c3.metric("Normal Predictions", f"{normal_count:,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Critical Risk", f"{critical_count:,}")
    c5.metric("High + Critical Risk", f"{high_critical_count:,}")
    c6.metric("Average Model Risk Score", f"{average_risk:.2f}/100")

    st.divider()
    st.subheader("Flagged Transaction Exposure")
    e1, e2 = st.columns(2)
    e1.metric("Total Value of Flagged Predictions", f"€{flagged_amount:,.2f}")
    average_flagged_amount = flagged_amount / flagged_count if flagged_count else 0
    e2.metric("Average Flagged Transaction Amount", f"€{average_flagged_amount:,.2f}")

    risk_data = (
        df["Risk_Category"].value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Transactions")
    )
    risk_data["Percentage"] = risk_data["Transactions"] / total_transactions * 100

    st.divider()
    st.subheader("Transaction Risk Distribution")
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            risk_data, x="Risk Category", y="Transactions",
            color="Risk Category", text="Transactions",
            color_discrete_map=RISK_COLORS,
            category_orders={"Risk Category": RISK_ORDER},
            title="Transactions by Risk Category",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)
    with right:
        fig = px.pie(
            risk_data, names="Risk Category", values="Transactions",
            hole=.60, color="Risk Category", color_discrete_map=RISK_COLORS,
            title="Risk Category Composition",
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)

    st.subheader("Risk Category Breakdown")
    st.dataframe(risk_data, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Transaction Amount Distribution")
    fig = px.histogram(
        df, x="Amount", nbins=50,
        labels={"Amount": "Transaction Amount (€)", "count": "Transactions"},
    )
    st.plotly_chart(style_plot(fig, 360), use_container_width=True)

    st.divider()
    st.subheader("Highest-Risk Flagged Transactions")
    suspicious = df[df["Predicted_Fraud"] == 1].sort_values("Risk_Score", ascending=False)
    if suspicious.empty:
        st.success("No transactions were flagged by the model.")
    else:
        columns = [c for c in ["Time", "Amount", "Fraud_Probability", "Risk_Score", "Risk_Category"] if c in suspicious.columns]
        st.dataframe(suspicious[columns].head(10), use_container_width=True, hide_index=True)

    st.info(
        "FraudGuard displays model-generated scores, not calibrated probabilities or confirmed fraud findings. "
        "Flagged transactions require human review."
    )


# ============================================================
# 7. TRANSACTION MONITORING
# ============================================================

elif page == "Transaction Monitoring":
    page_header("Transaction Monitoring", "Search, filter, sort and investigate model-scored transactions.")

    f1, f2, f3 = st.columns(3)
    with f1:
        search_query = st.text_input("Search dataset row index", placeholder="Enter row index")
    with f2:
        selected_risks = st.multiselect("Risk categories", RISK_ORDER, default=["High", "Critical"])
    with f3:
        decision_filter = st.selectbox("Model decision", ["All", "Flagged only", "Not flagged"])

    f4, f5 = st.columns(2)
    with f4:
        min_amount = st.number_input("Minimum transaction amount (€)", min_value=0.0, value=0.0, step=10.0)
    with f5:
        min_risk = st.slider("Minimum risk score", 0, 100, 0)

    filtered = df.copy()
    if search_query.strip():
        filtered = filtered[filtered.index.astype(str).str.contains(search_query.strip(), regex=False)]
    if selected_risks:
        filtered = filtered[filtered["Risk_Category"].isin(selected_risks)]
    else:
        filtered = filtered.iloc[0:0]
    filtered = filtered[(filtered["Amount"] >= min_amount) & (filtered["Risk_Score"] >= min_risk)]
    if decision_filter == "Flagged only":
        filtered = filtered[filtered["Predicted_Fraud"] == 1]
    elif decision_filter == "Not flagged":
        filtered = filtered[filtered["Predicted_Fraud"] == 0]

    sort_option = st.selectbox(
        "Sort results",
        ["Highest risk first", "Lowest risk first", "Highest amount first", "Lowest amount first"],
    )
    sort_map = {
        "Highest risk first": ("Risk_Score", False),
        "Lowest risk first": ("Risk_Score", True),
        "Highest amount first": ("Amount", False),
        "Lowest amount first": ("Amount", True),
    }
    sort_col, ascending = sort_map[sort_option]
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.subheader(f"{len(filtered):,} matching transactions")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching", f"{len(filtered):,}")
    m2.metric("Flagged", f"{int(filtered['Predicted_Fraud'].sum()):,}")
    m3.metric("Critical", f"{int((filtered['Risk_Category'] == 'Critical').sum()):,}")
    m4.metric("Transaction value", f"€{filtered['Amount'].sum():,.2f}")

    if filtered.empty:
        st.warning("No transactions match these filters. Try adjusting your selection.")
    else:
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (current_page - 1) * page_size
        page_data = filtered.iloc[start:start + page_size]
        st.caption(f"Showing {start + 1}–{min(start + page_size, len(filtered))} of {len(filtered):,}")
        st.dataframe(page_data, use_container_width=True, hide_index=False)
        download_csv(filtered, "Download filtered transactions", "fraudguard_filtered_transactions.csv", "monitor_download", include_index=True)

        st.divider()
        st.subheader("Transaction investigation")
        selected_index = st.selectbox(
            "Select a dataset row",
            filtered.index.tolist(),
            format_func=lambda value: f"Dataset row {value}",
            key="investigate_row",
        )
        selected = filtered.loc[selected_index]
        d1, d2, d3 = st.columns(3)
        d1.metric("Model risk score", f"{selected['Risk_Score']:.2f}/100")
        d2.metric("Transaction amount", f"€{selected['Amount']:,.2f}")
        d3.metric("Risk category", str(selected["Risk_Category"]))
        st.write("**Model decision:** " + ("Flagged for review" if selected["Predicted_Fraud"] == 1 else "Not flagged"))
        with st.expander("View all available fields for this transaction"):
            st.json(selected.to_dict())
        st.info("A model flag is not confirmation of fraud. Investigate before taking action.")


# ============================================================
# 8. RISK ANALYTICS
# ============================================================

elif page == "Risk Analytics":
    page_header("Risk Analytics", "Explore risk-score patterns, transaction values and available model evaluation data.")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Mean risk score", f"{df['Risk_Score'].mean():.2f}/100")
    q2.metric("Median risk score", f"{df['Risk_Score'].median():.2f}/100")
    q3.metric("95th percentile", f"{df['Risk_Score'].quantile(.95):.2f}/100")
    q4.metric("High + Critical", f"{high_critical_count:,}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="Risk_Score", nbins=50, title="Model Risk Score Distribution")
        fig.add_vline(x=threshold * 100, line_dash="dash", annotation_text="Decision threshold")
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)
    with right:
        fig = px.box(
            df, x="Risk_Category", y="Amount", color="Risk_Category",
            category_orders={"Risk_Category": RISK_ORDER},
            color_discrete_map=RISK_COLORS,
            title="Transaction Amount by Risk Category",
        )
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)

    summary = (
        df.groupby("Risk_Category", observed=False)
        .agg(
            Transactions=("Risk_Score", "size"),
            Average_Risk_Score=("Risk_Score", "mean"),
            Average_Amount=("Amount", "mean"),
            Total_Amount=("Amount", "sum"),
            Maximum_Risk_Score=("Risk_Score", "max"),
        )
        .reindex(RISK_ORDER)
        .fillna(0)
        .reset_index()
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            summary, x="Risk_Category", y="Average_Risk_Score",
            color="Risk_Category", color_discrete_map=RISK_COLORS,
            category_orders={"Risk_Category": RISK_ORDER},
            title="Average Risk Score by Category", text_auto=".2f",
        )
        st.plotly_chart(style_plot(fig, 380), use_container_width=True)
    with right:
        fig = px.bar(
            summary, x="Risk_Category", y="Total_Amount",
            color="Risk_Category", color_discrete_map=RISK_COLORS,
            category_orders={"Risk_Category": RISK_ORDER},
            title="Total Transaction Amount by Category",
        )
        st.plotly_chart(style_plot(fig, 380), use_container_width=True)

    st.subheader("Detailed risk summary")
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    download_csv(summary, "Download risk summary", "fraudguard_risk_summary.csv", "risk_summary_download")

    st.divider()
    st.subheader("Model evaluation")
    # Metrics are only calculated when actual ground-truth labels exist for these same rows.
    actual_column = next((c for c in ["Actual_Class", "Class"] if c in df.columns), None)
    if actual_column is None:
        st.info(
            "Ground-truth labels are not present in dashboard_transactions.csv. "
            "Accuracy, precision, recall, F1 and a confusion matrix are therefore not calculated. "
            "Do not use model predictions as actual labels."
        )
    else:
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            actual = pd.to_numeric(df[actual_column], errors="coerce")
            valid = actual.isin([0, 1])
            y_true = actual[valid].astype(int)
            y_pred = df.loc[valid, "Predicted_Fraud"].astype(int)
            if len(y_true) == 0:
                st.warning("No valid 0/1 ground-truth labels are available.")
            else:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                e2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
                e3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
                e4.metric("F1 score", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig = px.imshow(
                    cm, text_auto=True,
                    x=["Predicted normal", "Predicted fraud"],
                    y=["Actual normal", "Actual fraud"],
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                )
                st.plotly_chart(style_plot(fig, 360), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not calculate evaluation metrics: {exc}")

    st.caption("Risk scores are model estimates, not calibrated probabilities or confirmed fraud findings.")


# ============================================================
# 9. EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":
    page_header("Explainable AI", "Understand global feature importance and inspect individual model predictions.")

    st.info(
        "The dataset's V1–V28 columns are anonymized PCA-transformed features. "
        "They do not directly represent named financial behaviours."
    )

    st.subheader("Global feature importance")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        top_n = st.slider("Number of features", 5, len(importance), min(15, len(importance)))
        fig = px.bar(
            importance.head(top_n).sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title=f"Top {top_n} Random Forest Features",
        )
        st.plotly_chart(style_plot(fig, max(360, top_n * 27)), use_container_width=True)
        st.caption(
            "Random Forest impurity-based importance is a global diagnostic. "
            "It is not causal evidence and can be biased toward some feature types."
        )
    else:
        st.warning("This model does not expose feature_importances_.")

    st.divider()
    st.subheader("Explain one transaction")
    if not all(col in df.columns for col in feature_names):
        st.warning("The dashboard CSV does not contain all 30 model features. Individual explanations are unavailable from this export.")
    else:
        flagged_rows = df[df["Predicted_Fraud"] == 1]
        candidate_rows = flagged_rows if not flagged_rows.empty else df
        selected_index = st.selectbox(
            "Choose a transaction row",
            candidate_rows.index.tolist(),
            format_func=lambda value: f"Dataset row {value}",
            key="xai_transaction",
        )
        selected_row = candidate_rows.loc[[selected_index]]
        try:
            transaction_features = safe_numeric_frame(selected_row, feature_names)
            selected_score = float(model_scores(model, transaction_features)[0]) * 100
            st.metric("Model risk score", f"{selected_score:.2f}/100")
            with st.expander("View model input values"):
                st.dataframe(transaction_features.T.rename(columns={selected_index: "Value"}), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not score this transaction: {exc}")

        st.divider()
        st.subheader("SHAP explanation (optional)")
        st.write("SHAP provides a more detailed explanation of how features contribute to a model output.")
        if st.button("Generate SHAP explanation", type="primary"):
            try:
                import shap

                with st.spinner("Calculating SHAP values for a small sample..."):
                    # Limit the background and explanation sample to keep Colab resource use manageable.
                    background_n = min(40, len(df))
                    background = df[feature_names].sample(
                        n=background_n,
                        random_state=42,
                    )
                    background = safe_numeric_frame(background, feature_names)
                    explain_input = safe_numeric_frame(selected_row, feature_names)

                    explainer = shap.TreeExplainer(model, data=background)
                    shap_values = explainer.shap_values(explain_input)

                    # SHAP output format differs across versions and binary classifiers.
                    if isinstance(shap_values, list):
                        values = np.asarray(shap_values[-1])
                    else:
                        values = np.asarray(shap_values)
                        if values.ndim == 3:
                            if values.shape[-1] == 2:
                                values = values[:, :, 1]
                            elif values.shape[0] == 2:
                                values = values[1]
                        if values.ndim == 1:
                            values = values.reshape(1, -1)

                    if values.ndim != 2 or values.shape[1] != len(feature_names):
                        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

                    contribution = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP value": values[0],
                        "Absolute impact": np.abs(values[0]),
                    }).sort_values("Absolute impact", ascending=False)

                    st.success("SHAP explanation generated.")
                    fig = px.bar(
                        contribution.head(15).sort_values("SHAP value"),
                        x="SHAP value", y="Feature", orientation="h",
                        color="SHAP value",
                        color_continuous_scale="RdBu",
                        title="Feature contributions for selected transaction",
                    )
                    st.plotly_chart(style_plot(fig, 430), use_container_width=True)
                    st.dataframe(contribution.head(15), use_container_width=True, hide_index=True)
                    st.caption(
                        "SHAP values describe model output contributions relative to the chosen background sample. "
                        "They do not prove causation or fraud."
                    )
            except ImportError:
                st.error("SHAP is not installed. In Colab, run `!pip install shap`, then restart/re-run the app.")
            except Exception as exc:
                st.error(f"SHAP could not generate this explanation: {exc}")


# ============================================================
# 10. REPORTS & ALERTS
# ============================================================

elif page == "Reports & Alerts":
    page_header("Reports & Alerts", "Build review queues and export model-flagged transaction reports.")

    flagged = df[df["Predicted_Fraud"] == 1].sort_values("Risk_Score", ascending=False).copy()
    critical = df[df["Risk_Category"] == "Critical"].sort_values("Risk_Score", ascending=False).copy()
    high_critical = df[df["Risk_Category"].isin(["High", "Critical"])].sort_values("Risk_Score", ascending=False).copy()

    a, b, c = st.columns(3)
    a.metric("Flagged by model", f"{len(flagged):,}")
    b.metric("Critical risk", f"{len(critical):,}")
    c.metric("High + Critical", f"{len(high_critical):,}")

    st.subheader("Report settings")
    report_scope = st.selectbox(
        "Choose report",
        ["All model-flagged transactions", "High and Critical risk", "Critical risk only"],
    )
    if report_scope == "All model-flagged transactions":
        report_df = flagged
    elif report_scope == "High and Critical risk":
        report_df = high_critical
    else:
        report_df = critical

    st.write(f"**{len(report_df):,}** records in selected report")
    if report_df.empty:
        st.info("No records are available for this report.")
    else:
        columns = [c for c in ["Time", "Amount", "Fraud_Probability", "Risk_Score", "Risk_Category", "Predicted_Fraud"] if c in report_df.columns]
        st.dataframe(report_df[columns], use_container_width=True, hide_index=True)
        download_csv(report_df, "Download selected report (CSV)", "fraudguard_risk_report.csv", "report_download", include_index=True)

    st.divider()
    st.subheader("Risk summary report")
    summary = (
        df.groupby("Risk_Category", observed=False)
        .agg(
            Transactions=("Risk_Score", "size"),
            Average_Risk_Score=("Risk_Score", "mean"),
            Total_Transaction_Amount=("Amount", "sum"),
        )
        .reindex(RISK_ORDER)
        .fillna(0)
        .reset_index()
    )
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    download_csv(summary, "Download risk summary (CSV)", "fraudguard_risk_summary.csv", "report_summary_download")

    st.warning(
        "These reports contain model-generated review candidates, not confirmed fraud cases. "
        "Human investigation is required before any operational action."
    )


# ============================================================
# 11. UPLOAD & PREDICT
# ============================================================

elif page == "Upload & Predict":
    page_header("Upload & Predict", "Score new transactions with the saved Random Forest model.")

    st.info(
        "Upload a CSV containing the same 30 model input features used during training: "
        "Time, Amount and V1–V28. The Class target column is not required."
    )

    with st.expander("Required input columns"):
        st.code(", ".join(feature_names), language="text")
        st.caption("Column order can differ. FraudGuard selects the saved training order automatically.")

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"], key="upload_predict_csv")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        try:
            input_df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
            st.write(f"Uploaded file: **{len(input_df):,} rows × {len(input_df.columns)} columns**")

            if input_df.empty:
                st.error("The uploaded CSV contains no transaction rows.")
            else:
                missing = [feature for feature in feature_names if feature not in input_df.columns]
                if missing:
                    st.error("Missing required model features: " + ", ".join(missing))
                else:
                    input_data = safe_numeric_frame(input_df, feature_names)
                    if "Amount" in input_data.columns and (input_data["Amount"] < 0).any():
                        st.error("Transaction amounts cannot be negative.")
                    else:
                        st.success("File structure and numeric features validated.")
                        if st.button("Run fraud risk analysis", type="primary"):
                            with st.spinner("Scoring transactions..."):
                                scores = model_scores(model, input_data)
                                if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
                                    raise ValueError("The model returned invalid class scores.")

                                results = input_df.copy()
                                results["Model_Fraud_Score"] = scores
                                results["Risk_Score"] = np.round(scores * 100, 2)
                                results["Predicted_Fraud"] = (scores >= threshold).astype(int)
                                results["Risk_Category"] = [risk_category(float(x) * 100) for x in scores]
                                results["Review_Status"] = np.where(
                                    results["Predicted_Fraud"] == 1,
                                    "Flagged for human review",
                                    "Not flagged by model",
                                )

                                st.session_state["prediction_results"] = results
                                st.session_state["prediction_file_hash"] = file_hash

            if (
                "prediction_results" in st.session_state
                and st.session_state.get("prediction_file_hash") == file_hash
            ):
                results = st.session_state["prediction_results"]
                st.success("Prediction results are ready.")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Transactions analysed", f"{len(results):,}")
                p2.metric("Flagged", f"{int(results['Predicted_Fraud'].sum()):,}")
                p3.metric("Critical risk", f"{int((results['Risk_Category'] == 'Critical').sum()):,}")
                p4.metric("Average risk score", f"{results['Risk_Score'].mean():.2f}/100")

                st.subheader("Prediction results")
                st.dataframe(results, use_container_width=True, hide_index=True)
                download_csv(results, "Download prediction results", "fraudguard_prediction_results.csv", "prediction_download")

                st.warning(
                    "Model scores are estimates, not calibrated probabilities or confirmed fraud findings. "
                    "Flagged records should be reviewed by a human analyst."
                )
        except Exception as exc:
            st.error(f"Unable to process uploaded CSV: {exc}")


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()
st.caption(
    "FraudGuard | Academic ML Project | 2026  •  "
    "Model outputs support human review and do not establish confirmed fraud."
)
