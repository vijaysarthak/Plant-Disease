"""
streamlit_app.py — the FRONTEND: a web page where a farmer uploads a leaf photo and gets advice.

Run it from the project folder:
    streamlit run app/streamlit_app.py
The browser opens http://localhost:8501

How it is connected:
    this page  ->  src/predictor.py  ->  saved_model/ (made by notebooks/train_model.ipynb)
               ->  src/database.py   ->  database/predictions.db (history + feedback)

Streamlit re-runs this whole file from top to bottom every time the user clicks something.
That is why we remember results in `st.session_state` (a dictionary that survives the re-runs).
"""
import sys
from pathlib import Path

# Make the project folder importable (so "import config" and "from src ... import" work)
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import streamlit as st
from PIL import Image

from src import database
from src.disease_info import CLASS_NAMES, DISEASES, GROUP_CODES, GROUP_NAMES, pretty_class_name
from src.predictor import PlantDiseasePredictor

# ----------------------------------------------------------------------------
# Page settings and a little styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Plant Disease Advisor", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.result-card {padding: 1.1rem 1.3rem; border-radius: 12px; margin-bottom: 1rem; border-left: 8px solid;}
.advise  {background: #e9f7ef; border-color: #1e8449;}
.refer   {background: #fef5e7; border-color: #d68910;}
.unknown {background: #f2f3f4; border-color: #707b7c;}
.result-card h3 {margin: 0 0 .3rem 0;}
.chip {display: inline-block; padding: 2px 10px; margin: 2px; border-radius: 12px;
       background: #e8eaf6; color: #283593; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load the model ONCE (st.cache_resource keeps it in memory between re-runs)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return PlantDiseasePredictor()


try:
    predictor = get_predictor()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Open notebooks/train_model.ipynb, run all cells, then refresh this page.")
    st.stop()

database.create_tables()
info = predictor.info
scores = info.get("scores", {})

# ----------------------------------------------------------------------------
# Sidebar: information about the loaded model
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🌿 Plant Disease Advisor")
    st.caption("MCA Major Project MCA_MP_22 · Presidency University, Bengaluru")
    st.subheader("Loaded model")
    st.write(f"**Network:** {info['backbone']}" + (" (fine-tuned)" if info.get("finetuned") else " (frozen backbone)"))
    st.write(f"**Trained on:** {info['trained_on']}")
    if info.get("quick_test"):
        st.warning("This model was trained in QUICK_TEST mode (tiny data). Re-train with QUICK_TEST = False.")
    if scores:
        col1, col2 = st.columns(2)
        col1.metric("Lab accuracy", f"{scores['lab_accuracy']:.1%}")
        col2.metric("Field accuracy", f"{scores['field_accuracy']:.1%}")
        st.markdown("**Wrong advice on field photos**")
        col1, col2 = st.columns(2)
        col1.metric("This system", f"{scores['field_IAR']:.1%}")
        col2.metric("Normal top-1", f"{scores['field_IAR_top1']:.1%}")
        st.caption(f"Share of wrong treatment advice among the field photos (PlantDoc) where advice was given. "
                   f"This system gave advice for {scores['field_advice_rate']:.0%} of them and referred the rest.")
    st.subheader("Supported crops")
    crops = sorted(set(name.split("___")[0].replace("_", " ").replace(",", "") for name in CLASS_NAMES))
    st.write(", ".join(crops))

# ----------------------------------------------------------------------------
# Three tabs
# ----------------------------------------------------------------------------
tab_diagnose, tab_history, tab_how = st.tabs(["🔍 Diagnose a leaf", "🕘 History", "📖 How it works"])

# =============================== TAB 1: DIAGNOSE ===============================
with tab_diagnose:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("1. Upload a photo of ONE leaf")
        source = st.radio("Photo source", ["Upload a file", "Use the camera"], horizontal=True,
                          label_visibility="collapsed")
        if source == "Upload a file":
            photo_file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
        else:
            photo_file = st.camera_input("Take a photo")
        st.caption("Tip: daylight, sharp focus, the leaf filling most of the photo.")

        if photo_file is not None:
            image = Image.open(photo_file).convert("RGB")
            st.image(image, width="stretch")

    with right:
        st.subheader("2. Result")
        if photo_file is None:
            st.info("Upload a photo on the left to get a diagnosis.")
        else:
            # Predict only when a NEW photo arrives (not on every click)
            photo_id = f"{photo_file.name}-{photo_file.size}"
            if st.session_state.get("photo_id") != photo_id:
                result = predictor.predict(image)
                st.session_state["photo_id"] = photo_id
                st.session_state["result"] = result
                st.session_state["prediction_id"] = database.save_prediction(photo_file.name, result)
                st.session_state["feedback_sent"] = False
            result = st.session_state["result"]
            advice = result["advice"]

            # ---- the decision card ----
            if result["decision"] == "advise":
                heading = f"✅ Treatment group: {result['advised_group']}"
            elif result["decision"] == "refer":
                heading = "⚠️ Not sure — please ask an expert"
            else:
                heading = "❓ Photo not recognised"
            st.markdown(f"""
            <div class="result-card {result['decision']}">
              <h3>{heading}</h3>
              {advice['what_it_means']}
            </div>""", unsafe_allow_html=True)

            st.markdown("**What to do**")
            st.write(advice["what_to_do"])
            st.caption(f"Source: {advice['source']}")

            # ---- details ----
            with st.expander("Why this answer? (details)", expanded=result["decision"] != "advise"):
                chips = "".join(f'<span class="chip">{name}</span>' for name in result["prediction_set"]) or "—"
                st.markdown(f"**Groups the model cannot rule out (prediction set):** {chips}",
                            unsafe_allow_html=True)
                st.caption("One group → advice. Two or more → the treatments differ, so we refer to an expert.")

                st.markdown("**Probability of each treatment group**")
                for group_name, probability in result["group_probabilities"].items():
                    st.progress(min(probability, 1.0), text=f"{group_name}: {probability:.1%}")

                st.markdown("**Most likely diseases**")
                for item in result["top_classes"]:
                    st.write(f"- {item['pretty_name']} — {item['probability']:.1%}  "
                             f"*(cause: {item['cause']}, group: {item['group']})*")
                if result["is_unfamiliar"]:
                    st.write(f"Energy score {result['energy']:.2f} is above the limit "
                             f"{predictor.energy_threshold:.2f}: the photo looks unlike the training photos.")

            # ---- feedback ----
            st.subheader("3. Was this right?")
            if st.session_state["feedback_sent"]:
                st.success("Thank you! Your feedback was saved.")
            else:
                with st.form("feedback"):
                    answer = st.radio("Answer", ["Yes", "No", "I don't know"], horizontal=True)
                    true_class = st.selectbox("If you know the real disease, choose it (optional)",
                                              ["—"] + CLASS_NAMES, format_func=lambda n: n if n == "—" else pretty_class_name(n))
                    comment = st.text_input("Comment (optional)")
                    if st.form_submit_button("Send feedback"):
                        if answer != "I don't know":
                            database.save_feedback(st.session_state["prediction_id"], answer == "Yes",
                                                   "" if true_class == "—" else true_class, comment)
                        st.session_state["feedback_sent"] = True
                        st.rerun()

# =============================== TAB 2: HISTORY ===============================
with tab_history:
    st.subheader("Last 50 predictions")
    history = database.get_history(limit=50)
    if len(history) == 0:
        st.info("No predictions yet.")
    else:
        table = pd.DataFrame(history)
        table["top_class"] = table["top_class"].map(pretty_class_name)
        table["top_probability"] = (table["top_probability"] * 100).round(1).astype(str) + " %"
        table["is_correct"] = table["is_correct"].map({1: "✅ yes", 0: "❌ no"}).fillna("")
        table = table.rename(columns={"id": "#", "created_at": "time", "image_name": "photo",
                                      "top_class": "most likely disease", "top_probability": "probability",
                                      "decision": "decision", "advised_group": "advised group",
                                      "prediction_set": "prediction set", "is_correct": "feedback"})
        st.dataframe(table, width="stretch", hide_index=True)

# =============================== TAB 3: HOW IT WORKS ===============================
with tab_how:
    st.subheader("How the advisor decides")
    st.markdown("""
1. **The network** (trained in `notebooks/train_model.ipynb`) gives a probability for each of 38 diseases.
2. The diseases are put into **6 treatment groups** — what the farmer has to *do*.
   The probability of a group = sum of its diseases' probabilities.
3. Every group has a **threshold** learned from labelled field photos (*conformal prediction*).
   All groups that pass are the **prediction set**. It contains the correct group about 90 % of the time.
4. **Decision:**
   * exactly **one** group → give that group's advice
   * **two or more** groups → the treatments would differ → **refer to an expert (KVK)**
   * the photo looks unlike anything seen in training (high *energy* score) → **photo not recognised**

This is safer than trusting the single most likely disease: on field photos a normal classifier is often
wrong while being confident.
""")
    group_table = []
    for code in GROUP_CODES:
        members = [pretty_class_name(n) for n in CLASS_NAMES if DISEASES[n][1] == code]
        group_table.append({"group": f"{code} {GROUP_NAMES[code]}", "diseases": ", ".join(members)})
    st.table(pd.DataFrame(group_table))
    st.caption("⚠️ This tool supports, but does not replace, an agriculture expert. "
               "Always follow the label of a registered (CIB&RC) product.")
