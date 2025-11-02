import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Code → meaning (only used for Info tab) ---
CODE_LABELS = {
    "cap-shape": {"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"},
    "cap-surface": {"f":"fibrous","g":"grooves","y":"scaly","s":"smooth"},
    "cap-color": {"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"},
    "bruises": {"t":"bruises","f":"no"},
    "odor": {"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"},
    "gill-attachment": {"a":"attached","d":"descending","f":"free","n":"notched"},
    "gill-spacing": {"c":"close","w":"crowded","d":"distant"},
    "gill-size": {"b":"broad","n":"narrow"},
    "gill-color": {"k":"black","n":"brown","b":"buff","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"},
    "stalk-shape": {"e":"enlarging","t":"tapering"},
    "stalk-root": {"b":"bulbous","c":"club","u":"cup","e":"equal","z":"rhizomorphs","r":"rooted"},  # '?' = missing
    "stalk-surface-above-ring": {"f":"fibrous","y":"scaly","k":"silky","s":"smooth"},
    "stalk-surface-below-ring": {"f":"fibrous","y":"scaly","k":"silky","s":"smooth"},
    "stalk-color-above-ring": {"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"},
    "stalk-color-below-ring": {"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"},
    "veil-type": {"p":"partial","u":"universal"},
    "veil-color": {"n":"brown","o":"orange","w":"white","y":"yellow"},
    "ring-number": {"n":"none","o":"one","t":"two"},
    "ring-type": {"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"},
    "spore-print-color": {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"},
    "population": {"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"},
    "habitat": {"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"},
}

st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄")
st.title("🍄 Mushroom Classifier")
st.caption("Predict **edible vs poisonous** from categorical features.\n"
           "⚠️ Educational demo only — do **NOT** use for real-life decisions.")

APP_DIR = Path(__file__).parent.resolve()
MODEL_PATH = APP_DIR / "model.joblib"

if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Run `python train.py` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
columns = bundle["columns"]
cat_levels = bundle["cat_levels"]
target_names = bundle["target_names"]  # ["edible","poisonous"]

# --- Two tabs: Predict | Info ---
tab_pred, tab_info = st.tabs(["🔮 Predict", "ℹ️ Info"])

with tab_pred:
    st.header("Single prediction")
    with st.form("single_pred"):
        c1, c2 = st.columns(2)
        inputs = {}
        for i, col in enumerate(columns):
            # keep dropdown letters for simplicity
            options = ["(unknown)"] + sorted(list(cat_levels[col]))
            with (c1 if i % 2 == 0 else c2):
                sel = st.selectbox(col, options, index=0, key=f"sb_{col}")
                inputs[col] = None if sel == "(unknown)" else sel

        submitted = st.form_submit_button("Predict")
        if submitted:
            X = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in inputs.items()}])[columns]
            proba = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            st.subheader(f"Prediction: **{target_names[pred_idx]}**")
            st.write({target_names[0]: float(proba[0]), target_names[1]: float(proba[1])})

    st.header("Batch prediction (CSV)")
    st.write("Upload a CSV with **exactly these columns** (use the codes; no `class`):")
    st.code(", ".join(columns), language="text")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df = df[columns].replace({"(unknown)": np.nan})
                preds = model.predict(df)
                probas = model.predict_proba(df)
                out = df.copy()
                out["prediction"] = [target_names[i] for i in preds]
                out["proba_edible"] = probas[:, 0]
                out["proba_poisonous"] = probas[:, 1]
                st.success("Predictions complete.")
                st.dataframe(out.head(25))
                st.download_button(
                    "Download predictions as CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="mushroom_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.exception(e)

with tab_info:
    st.header("What the letters mean")
    st.write("This tab lists the **code → meaning** for every feature. `(unknown)` corresponds to missing values.")
    for col in columns:
        mapping = CODE_LABELS.get(col, {})
        if not mapping:
            continue
        st.subheader(col)
        # simple 2-column table
        table = pd.DataFrame(
            [{"code": k, "meaning": v} for k, v in sorted(mapping.items(), key=lambda kv: kv[1].lower())]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Source: UCI Mushroom dataset (tabular features).")
