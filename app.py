"""
Student Success Predictor  -  Streamlit app (red / black / purple theme)

Run:   streamlit run app.py
Needs: dropout_model.pkl (created by the last notebook cell) in the same folder.
Tip:   put .streamlit/config.toml next to this file for the full dark theme.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------- palette
BG, PANEL, LINE = "#0A0A0B", "#141416", "#2A2A2E"
TEXT, MUTED = "#F4F4F5", "#8B8B94"
RED, PURPLE = "#FF4D5E", "#A78BFA"
CLASS_COLORS = {"Dropout": RED, "Enrolled": MUTED, "Graduate": PURPLE}

MODEL_PATH = Path(__file__).parent / "dropout_model.pkl"

st.markdown(
    f"""
<style>
.stApp {{ background:{BG}; color:{TEXT}; }}
section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.4rem; }}
.app-title {{ font-size:2rem; font-weight:700; color:{PURPLE}; margin:0; }}
.app-sub {{ color:{MUTED}; margin:2px 0 14px 0; }}
.card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:14px 16px; }}
.card .lab {{ color:{MUTED}; font-size:.78rem; margin:0; }}
.card .val {{ font-size:1.7rem; font-weight:700; margin:2px 0 0 0; }}
.banner {{ border-radius:12px; padding:12px 18px; margin:6px 0 14px 0; font-size:1.02rem; }}
.pill {{ display:inline-block; padding:2px 12px; border-radius:999px; font-weight:600;
         font-size:.82rem; color:#0A0A0B; }}
.explain {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px;
            padding:10px 14px; margin:6px 0 16px 0; font-size:.88rem; color:#D4D4D8; }}
.explain b {{ color:{PURPLE}; }}
h2, h3 {{ color:{PURPLE} !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- load model
if not MODEL_PATH.exists():
    st.error("dropout_model.pkl was not found. Run the last notebook cell and "
             "place the file in the same folder as app.py.")
    st.stop()


@st.cache_resource
def load_artifact():
    # Only open pickle files you created yourself: pickle can execute code.
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(load_artifact()["model"])


art = load_artifact()
MODEL = art["model"]
FEATURES = list(art["features"])
CLASSES = list(art["class_names"])
DEFAULTS = art["defaults"]
RANGES = art["ranges"]
METRICS = art.get("metrics", {})
DROP = CLASSES.index("Dropout")


# ----------------------------------------------------------------- helpers
def predict_proba(df):
    return MODEL.predict_proba(df[FEATURES])


def shap_for_class(row_df, cls):
    """SHAP values of one row for one class (works for old and new SHAP)."""
    sv = get_explainer().shap_values(row_df[FEATURES])
    if isinstance(sv, list):
        return np.asarray(sv[cls])[0]
    sv = np.asarray(sv)
    return sv[0, :, cls] if sv.ndim == 3 else sv[0]


def risk_level(p):
    if p >= 0.60:
        return "High", RED
    if p >= 0.35:
        return "Medium", "#F59E0B"
    return "Low", "#34D399"


def card(label, value, color=TEXT):
    return (f'<div class="card"><p class="lab">{label}</p>'
            f'<p class="val" style="color:{color}">{value}</p></div>')


def explain(text):
    """A small 'how to read this' box under every chart."""
    st.markdown(f'<div class="explain"><b>How to read this.</b> {text}</div>',
                unsafe_allow_html=True)


def style_fig(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def num_slider(label, col, is_int=True, step=0.5):
    lo, hi = RANGES[col]
    d = DEFAULTS[col]
    if is_int:
        lo, hi, d = int(lo), int(hi), int(round(d))
        if hi <= lo:
            hi = lo + 1
        d = min(max(d, lo), hi)
        return st.slider(label, lo, hi, d, key=col)
    lo, hi, d = float(lo), float(hi), float(d)
    if hi <= lo:
        hi = lo + 1.0
    d = min(max(d, lo), hi)
    return st.slider(label, lo, hi, float(round(d, 1)), step=float(step), key=col)


def yes_no(label, col):
    idx = 1 if round(DEFAULTS[col]) >= 1 else 0
    return 1 if st.radio(label, ["No", "Yes"], index=idx, horizontal=True, key=col) == "Yes" else 0


def recommendations(v):
    tips = []
    if v["Curricular units 1st sem (approved)"] <= 2 or v["Curricular units 2nd sem (approved)"] <= 2:
        tips.append("Very few course units approved: arrange tutoring or a study-plan meeting.")
    g2 = v["Curricular units 2nd sem (grade)"]
    if 0 < g2 < 10:
        tips.append("Low 2nd-semester grades: offer mentoring and early feedback on assessments.")
    if v["Tuition fees up to date"] == 0 or v["Debtor"] == 1:
        tips.append("Financial strain detected: connect the student with the financial-aid office.")
    if v["Curricular units 2nd sem (enrolled)"] < v["Curricular units 1st sem (enrolled)"]:
        tips.append("Fewer units enrolled in semester 2: ask about workload, health or personal reasons.")
    if not tips:
        tips.append("No obvious warning signs in these inputs. Keep monitoring progress each semester.")
    return tips


# ----------------------------------------------------------------- sidebar
st.sidebar.markdown("<h3 style='margin-bottom:0'>Student profile</h3>", unsafe_allow_html=True)
st.sidebar.caption("Change the values. Every feature not shown here uses the training-data median.")
values = dict(DEFAULTS)

with st.sidebar.expander("Academic record", expanded=True):
    values["Admission grade"] = num_slider("Admission grade", "Admission grade", False, 0.5)
    values["Previous qualification (grade)"] = num_slider(
        "Previous qualification grade", "Previous qualification (grade)", False, 0.5)
    st.markdown("**1st semester**")
    values["Curricular units 1st sem (enrolled)"] = num_slider("Units enrolled", "Curricular units 1st sem (enrolled)")
    values["Curricular units 1st sem (approved)"] = num_slider("Units approved", "Curricular units 1st sem (approved)")
    values["Curricular units 1st sem (grade)"] = num_slider("Average grade", "Curricular units 1st sem (grade)", False, 0.1)
    st.markdown("**2nd semester**")
    values["Curricular units 2nd sem (enrolled)"] = num_slider("Units enrolled ", "Curricular units 2nd sem (enrolled)")
    values["Curricular units 2nd sem (approved)"] = num_slider("Units approved ", "Curricular units 2nd sem (approved)")
    values["Curricular units 2nd sem (grade)"] = num_slider("Average grade ", "Curricular units 2nd sem (grade)", False, 0.1)

with st.sidebar.expander("Personal"):
    values["Age at enrollment"] = num_slider("Age at enrollment", "Age at enrollment")
    gender_idx = 1 if round(DEFAULTS["Gender"]) >= 1 else 0
    values["Gender"] = 1 if st.radio("Gender", ["Female", "Male"], index=gender_idx,
                                     horizontal=True, key="gender") == "Male" else 0
    values["International"] = yes_no("International student", "International")
    values["Displaced"] = yes_no("Displaced from home", "Displaced")

with st.sidebar.expander("Financial"):
    values["Scholarship holder"] = yes_no("Scholarship holder", "Scholarship holder")
    values["Debtor"] = yes_no("Debtor", "Debtor")
    values["Tuition fees up to date"] = yes_no("Tuition fees up to date", "Tuition fees up to date")

student = pd.DataFrame([values])[FEATURES].astype(float)
proba = predict_proba(student)[0]
pred_idx = int(np.argmax(proba))
pred_name = CLASSES[pred_idx]
p_drop = float(proba[DROP])
level, level_color = risk_level(p_drop)

# ----------------------------------------------------------------- header
st.markdown('<p class="app-title">Student Success Predictor</p>'
            '<p class="app-sub">An early-warning model that estimates whether a student will drop out, '
            'stay enrolled or graduate, and explains why. XGBoost + SHAP.</p>',
            unsafe_allow_html=True)

tab_pred, tab_explain, tab_whatif, tab_cohort, tab_about = st.tabs(
    ["Prediction", "Explanation", "What-if", "Cohort dashboard", "About the model"])

# ================================================================= PREDICTION
with tab_pred:
    st.markdown(
        f'<div class="banner" style="background:{level_color}22;border:1px solid {level_color}">'
        f'Predicted outcome: <b style="color:{CLASS_COLORS.get(pred_name, TEXT)}">{pred_name}</b>'
        f' &nbsp;|&nbsp; Dropout risk: <span class="pill" style="background:{level_color}">'
        f'{level} ({p_drop:.0%})</span></div>', unsafe_allow_html=True)

    explain(
        f"The model looked at the {len(FEATURES)} facts about this student and estimates a "
        f"<b>{p_drop:.0%}</b> chance of dropping out. <b>Predicted outcome</b> is simply the class "
        f"with the highest probability. <b>Risk level</b> is based on the dropout probability: "
        f"below 35% is Low, 35 to 60% is Medium, 60% or more is High. This is an estimate based on "
        f"patterns in past students, not a certainty.")

    k = st.columns(4)
    k[0].markdown(card("Dropout risk", f"{p_drop:.0%}", RED), unsafe_allow_html=True)
    k[1].markdown(card("Predicted outcome", pred_name, CLASS_COLORS.get(pred_name, TEXT)), unsafe_allow_html=True)
    k[2].markdown(card("2nd sem units approved", int(values["Curricular units 2nd sem (approved)"])), unsafe_allow_html=True)
    k[3].markdown(card("2nd sem average grade", f"{values['Curricular units 2nd sem (grade)']:.1f}"), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p_drop * 100,
            number={"suffix": "%", "font": {"size": 44, "color": TEXT}},
            title={"text": "Dropout probability"},
            gauge={"axis": {"range": [0, 100], "tickcolor": MUTED},
                   "bar": {"color": RED, "thickness": 0.3},
                   "bgcolor": LINE, "borderwidth": 0,
                   "steps": [{"range": [0, 35], "color": "#1B1B1F"},
                             {"range": [35, 60], "color": "#242429"},
                             {"range": [60, 100], "color": "#2E1A1E"}]}))
        st.plotly_chart(style_fig(gauge, 320), key="gauge")
        explain("The red bar shows the estimated dropout probability from 0 to 100%. "
                "The three dark zones mark Low (0 to 35), Medium (35 to 60) and High (60 to 100) risk.")
    with c2:
        bars = go.Figure(go.Bar(
            x=[float(p) * 100 for p in proba], y=CLASSES, orientation="h",
            marker_color=[CLASS_COLORS.get(c, MUTED) for c in CLASSES],
            text=[f"{float(p):.1%}" for p in proba], textposition="auto"))
        bars.update_layout(title="Class probabilities", xaxis=dict(range=[0, 100], title="Probability (%)"))
        st.plotly_chart(style_fig(bars, 320), key="classbars")
        explain("Each bar is the model's probability for one outcome. The three bars always add up to "
                "100%. A long red bar means the model leans towards dropout; a long purple bar means graduation.")

    st.markdown("### Suggested support actions")
    for t in recommendations(values):
        st.markdown(f"- {t}")
    st.caption("A high-risk flag should trigger support for the student, never penalties or restrictions.")

# ================================================================= EXPLANATION
with tab_explain:
    st.markdown("### Why did the model predict this?")
    choice = st.selectbox("Explain the probability of", CLASSES, index=pred_idx, key="shap_class")
    ci = CLASSES.index(choice)
    sv = shap_for_class(student, ci)
    top = np.argsort(np.abs(sv))[::-1][:12][::-1]
    labels = [f"{FEATURES[i]} = {student.iloc[0, i]:.4g}" for i in top]
    fig = go.Figure(go.Bar(x=sv[top], y=labels, orientation="h",
                           marker_color=[RED if v > 0 else PURPLE for v in sv[top]]))
    fig.update_layout(title=f"Feature impact on '{choice}'", xaxis_title="SHAP value (impact on the model output)")
    st.plotly_chart(style_fig(fig, 520), key="shapbar")
    explain(f"Each bar is one fact about the student. <b>Red bars push the prediction towards "
            f"{choice}</b>, <b>purple bars push it away</b>. The longer the bar, the stronger the push. "
            f"These are SHAP values: they split the model's final score into a contribution from each feature. "
            f"The units are the model's internal log-odds scale, so compare bar lengths with each other "
            f"rather than reading them as percentages.")

    st.markdown("### In plain words")
    for i in np.argsort(np.abs(sv))[::-1][:3]:
        direction = "increases" if sv[i] > 0 else "decreases"
        st.markdown(f"- **{FEATURES[i]} = {student.iloc[0, i]:.4g}** {direction} the estimate for {choice}.")

# ================================================================= WHAT-IF
with tab_whatif:
    st.markdown("### What happens if one factor changes?")
    options = ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
               "Curricular units 2nd sem (grade)", "Curricular units 1st sem (grade)",
               "Admission grade", "Age at enrollment"]
    feat = st.selectbox("Feature to vary", options, index=1, key="whatif_feat")
    lo, hi = RANGES[feat]
    grid = np.linspace(float(lo), float(hi), 40)
    sweep = pd.concat([student] * len(grid), ignore_index=True)
    sweep[feat] = grid
    pr = predict_proba(sweep)

    fig = go.Figure()
    for i, c in enumerate(CLASSES):
        fig.add_trace(go.Scatter(x=grid, y=pr[:, i] * 100, mode="lines", name=c,
                                 line=dict(color=CLASS_COLORS.get(c, MUTED), width=3)))
    fig.add_vline(x=float(values[feat]), line_dash="dash", line_color=MUTED)
    fig.update_layout(title=f"Probability vs. {feat}", xaxis_title=feat, yaxis_title="Probability (%)")
    st.plotly_chart(style_fig(fig, 430), key="whatif")
    explain("Each line is the probability of one outcome as the chosen feature moves from its lowest to its "
            "highest value, while every other input stays as set in the sidebar. The dashed vertical line is "
            "this student's current value. A steep red line means the dropout estimate is very sensitive to "
            "this feature. This shows what the model learned, which is a pattern in the data and not proof "
            "that changing the feature would cause a different real outcome.")

# ================================================================= COHORT DASHBOARD
with tab_cohort:
    st.markdown("### Model performance on held-out test students")
    m = st.columns(4)
    m[0].markdown(card("Test accuracy", f"{METRICS.get('accuracy', 0):.1%}"), unsafe_allow_html=True)
    m[1].markdown(card("Test macro-F1", f"{METRICS.get('macro_f1', 0):.3f}", PURPLE), unsafe_allow_html=True)
    m[2].markdown(card("Training students", METRICS.get("n_train", "n/a")), unsafe_allow_html=True)
    m[3].markdown(card("Features", len(FEATURES)), unsafe_allow_html=True)
    st.write("")

    cm = art.get("confusion_matrix")
    if cm is not None:
        cm = np.array(cm)
        heat = go.Figure(go.Heatmap(
            z=cm, x=CLASSES, y=CLASSES, text=cm, texttemplate="%{text}",
            colorscale=[[0, "#1A0F11"], [1, RED]], showscale=False,
            textfont=dict(size=16, color=TEXT)))
        heat.update_layout(title="Confusion matrix (test set)",
                           xaxis_title="Predicted", yaxis_title="Actual",
                           yaxis=dict(autorange="reversed"))
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(style_fig(heat, 380), key="cm")
        with right:
            pc = art.get("per_class")
            if pc:
                rows = [{"Class": c, "Precision": round(pc[c]["precision"], 3),
                         "Recall": round(pc[c]["recall"], 3),
                         "F1": round(pc[c]["f1-score"], 3),
                         "Students": int(pc[c]["support"])} for c in CLASSES if c in pc]
                st.markdown("**Per-class scores**")
                st.dataframe(pd.DataFrame(rows), hide_index=True)
        explain("Rows are the <b>actual</b> outcome and columns are what the model <b>predicted</b>. "
                "The diagonal (top-left to bottom-right) counts correct predictions, so you want those cells "
                "to be bright red. Off-diagonal cells are mistakes. Example: the cell at row Dropout, column "
                "Graduate counts students who dropped out but were predicted to graduate. "
                "<b>Precision</b> = of the students predicted as a class, how many really were. "
                "<b>Recall</b> = of the students who really were that class, how many the model found. "
                "<b>F1</b> balances the two, and <b>macro-F1</b> averages F1 over the three classes equally.")
    else:
        st.info("Re-run the updated notebook cell to save the confusion matrix into the pickle file.")

    st.markdown("### Score a whole cohort")
    st.caption("Upload a CSV with the same columns as the original dataset (the original data.csv works; "
               "a Target column is ignored).")
    up = st.file_uploader("CSV file", type="csv", key="cohort_upload")
    if up is not None:
        try:
            raw = pd.read_csv(up, sep=None, engine="python")
            raw.columns = raw.columns.str.strip()
            missing = [c for c in FEATURES if c not in raw.columns]
            if missing:
                st.error(f"The file is missing {len(missing)} required columns, for example: {missing[:5]}")
            else:
                X = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
                X = X.fillna(pd.Series(DEFAULTS)).astype(float)
                P = predict_proba(X)
                pdrop = P[:, DROP]
                out = raw.copy()
                out["Predicted"] = [CLASSES[i] for i in P.argmax(1)]
                for i, c in enumerate(CLASSES):
                    out[f"P({c})"] = P[:, i].round(3)

                kk = st.columns(4)
                kk[0].markdown(card("Students scored", f"{len(out):,}"), unsafe_allow_html=True)
                kk[1].markdown(card("High risk (60%+)", f"{int((pdrop >= 0.6).sum()):,}", RED), unsafe_allow_html=True)
                kk[2].markdown(card("Average dropout probability", f"{pdrop.mean():.0%}"), unsafe_allow_html=True)
                n_grad = int((P.argmax(1) == CLASSES.index("Graduate")).sum())
                kk[3].markdown(card("Predicted to graduate", f"{n_grad:,}", PURPLE), unsafe_allow_html=True)
                st.write("")

                a, b = st.columns(2)
                with a:
                    counts = out["Predicted"].value_counts().reindex(CLASSES).fillna(0)
                    f1 = go.Figure(go.Bar(x=list(counts.index), y=counts.values,
                                          marker_color=[CLASS_COLORS.get(c, MUTED) for c in counts.index]))
                    f1.update_layout(title="Predicted outcomes", yaxis_title="Students")
                    st.plotly_chart(style_fig(f1, 320), key="cohort_counts")
                    explain("How many students the model puts in each outcome class. A tall red bar means "
                            "many students are predicted to drop out.")
                with b:
                    ages = pd.cut(X["Age at enrollment"], [0, 20, 25, 30, 200],
                                  labels=["20 or under", "21-25", "26-30", "30+"])
                    by_age = pd.Series(pdrop * 100).groupby(ages.values, observed=False).mean()
                    f2 = go.Figure(go.Bar(x=[str(i) for i in by_age.index],
                                          y=by_age.fillna(0).values, marker_color=RED))
                    f2.update_layout(title="Average dropout probability by age group",
                                     yaxis_title="Dropout probability (%)")
                    st.plotly_chart(style_fig(f2, 320), key="cohort_age")
                    explain("Each bar is the average predicted dropout probability for one age group. "
                            "Large differences between groups are worth investigating: they can reflect "
                            "real needs, or bias in the data. Check this before acting on predictions.")

                st.markdown("### Highest-risk students")
                order = np.argsort(-pdrop)[:10]
                top_rows = []
                for r in order:
                    sv_r = shap_for_class(X.iloc[[int(r)]], DROP)
                    driver_i = int(np.argmax(sv_r))
                    top_rows.append({
                        "Row": int(r) + 1,
                        "Dropout probability (%)": round(float(pdrop[r]) * 100, 1),
                        "Top driver": f"{FEATURES[driver_i]} = {X.iloc[int(r), driver_i]:.4g}"})
                st.dataframe(pd.DataFrame(top_rows), hide_index=True)
                explain("The ten students with the highest dropout probability. <b>Top driver</b> is the single "
                        "feature that pushed that student's estimate up the most (from SHAP), which tells an "
                        "advisor where to start the conversation. Use it to offer help, not to penalise.")

                st.download_button("Download all predictions (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv", key="dl")
        except Exception as e:
            st.error(f"Could not process this file: {e}")

# ================================================================= ABOUT
with tab_about:
    st.markdown("### What drives the model overall")
    imp = pd.Series(MODEL.feature_importances_, index=FEATURES).sort_values().tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=list(imp.index), orientation="h", marker_color=PURPLE))
    fig.update_layout(title="Top 15 global feature importances", xaxis_title="Importance")
    st.plotly_chart(style_fig(fig, 480), key="imp")
    explain("Longer bars mean the model relied more on that feature across all students. This is a global "
            "view; the Explanation tab shows the same idea for one student. Importance shows what the model "
            "uses, not what causes dropout.")

    st.markdown("### Glossary")
    st.markdown(
        "- **Macro-F1**: average of the F1 score of each class, so small classes count as much as large ones.\n"
        "- **SHAP**: a method that splits a prediction into a contribution from each feature.\n"
        "- **Recall for Dropout**: of the students who really dropped out, the share the model caught.\n"
        "- **Confusion matrix**: a table of actual vs. predicted classes.")

    st.markdown("### Responsible use")
    st.markdown(
        "- This is decision support, not a decision-maker. Use flags to offer help early.\n"
        "- Check performance across groups (gender, age, scholarship status) before real deployment.\n"
        "- Predictions reflect one institution's data and may not transfer elsewhere.")
    st.caption("Data: Realinho, V. et al. (2021). Predict Students' Dropout and Academic Success. "
               "UCI Machine Learning Repository, CC BY 4.0.")
