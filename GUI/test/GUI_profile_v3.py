#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations
from pathlib import Path
import string
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Oil Mk2 – Strain Page", page_icon="🧫", layout="wide")

DATA_PATH = Path(r"C:\Users\frodr\Documents\Strain Vaults\Notebooks\ML\Task_Models\tm_oil\DL\kfold\Mk2\runs\oil_hybrid_concat_v0_fold1\test_probas.csv")   # adjust if needed
IMAGES_DIR = Path("images")           # optional: images/<strain_id>.png|jpg|jpeg|webp

CLASS_COLS = ["A_Alkanes", "C_Aromatics", "D_Biosurfactants"]
CLASS_LABELS = {
    "A_Alkanes": "Alkanes",
    "C_Aromatics": "Aromatics",
    "D_Biosurfactants": "Biosurfactants",
}

# -------------------------
# Data
# -------------------------
@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "strain_id"})
    df["strain_id"] = df["strain_id"].astype(str)
    for c in CLASS_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[CLASS_COLS] = df[CLASS_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df

df = load_data(DATA_PATH)

def find_image(strain_id: str) -> Path | None:
    if not IMAGES_DIR.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = IMAGES_DIR / f"{strain_id}{ext}"
        if p.exists():
            return p
    return None

def goto_strain(sid: str):
    st.query_params["strain"] = sid

def permalink(sid: str) -> str:
    return f"?strain={sid}"

# -------------------------
# Routing (single-strain only)
# -------------------------
q = st.query_params
requested_id = q.get("strain", None)
if isinstance(requested_id, list):
    requested_id = requested_id[0]

st.title("Oil Mk2 – Strain Page")

# -------------------------
# Landing view: search + clickable glossary
# -------------------------
if not requested_id:
    st.info("Enter a strain ID or click one from the glossary to open its dedicated page. This app never lists full details for multiple strains on the same screen.")

    # Search box (exact match)
    sid = st.text_input("Strain ID", value="", placeholder="Exact strain ID…").strip()
    colA, colB = st.columns([1, 3])
    with colA:
        submit = st.button("Open")
    if (submit or sid) and sid:
        if (df["strain_id"] == sid).any():
            goto_strain(sid)
            st.stop()
        else:
            st.error(f"Strain '{sid}' not found in dataset.")

    st.divider()
    st.subheader("Glossary of strains")

    # A–Z letter filter
    all_ids = sorted(df["strain_id"].unique(), key=lambda x: x.lower())
    # Determine active letter from query param "letter" (optional nicety)
    active_letter = st.query_params.get("letter", None)
    if isinstance(active_letter, list):
        active_letter = active_letter[0]
    letters = list(string.ascii_uppercase) + ["#"]  # "#" = non A–Z

    # Render letter buttons
    btn_cols = st.columns(14)
    for i, L in enumerate(letters):
        with btn_cols[i % 14]:
            clicked = st.button(L, use_container_width=True)
            if clicked:
                st.query_params["letter"] = L
                st.experimental_rerun()

    # Filter IDs by letter (if chosen)
    def starts_with_letter(s: str, L: str) -> bool:
        if not s:
            return False
        head = s[0].upper()
        if L == "#":
            return head not in string.ascii_uppercase
        return head == L

    filtered_ids = all_ids
    if active_letter:
        filtered_ids = [sid for sid in all_ids if starts_with_letter(sid, active_letter)]

    # Multi-column clickable list (no heavy table to keep UI snappy)
    if not filtered_ids:
        st.caption("No strains under this letter.")
    else:
        N_COLS = 4
        cols = st.columns(N_COLS, gap="small")
        for i, sid in enumerate(filtered_ids):
            with cols[i % N_COLS]:
                st.markdown(f"- [{sid}]({permalink(sid)})")

    st.stop()

# -------------------------
# Single-strain page
# -------------------------
row = df.loc[df["strain_id"] == requested_id]
if row.empty:
    st.error(f"Strain '{requested_id}' not found.")
    st.stop()

r = row.iloc[0]
st.markdown(f"### Strain: `{r['strain_id']}`")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Alkanes", f"{r['A_Alkanes']:.2%}")
with c2:
    st.metric("Aromatics", f"{r['C_Aromatics']:.2%}")
with c3:
    st.metric("Biosurfactants", f"{r['D_Biosurfactants']:.2%}")

img_path = find_image(r["strain_id"])
if img_path:
    st.image(str(img_path), caption=f"Strain image – {r['strain_id']}", use_container_width=True)
else:
    st.caption("No image found. Add one at `images/<strain_id>.jpg|png|jpeg|webp`.")

st.markdown("#### Class probabilities")
prob_df = (pd.Series(
    {
        CLASS_LABELS["A_Alkanes"]: float(r["A_Alkanes"]),
        CLASS_LABELS["C_Aromatics"]: float(r["C_Aromatics"]),
        CLASS_LABELS["D_Biosurfactants"]: float(r["D_Biosurfactants"]),
    }
).to_frame("probability").sort_values("probability", ascending=False))
st.bar_chart(prob_df)

with st.expander("Raw data"):
    st.json(r.to_dict())

single_csv = row.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download this strain (CSV)",
    data=single_csv,
    file_name=f"{r['strain_id']}_oil_mk2_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)

# Quick navigation: go back to glossary or jump directly
left, right = st.columns([1,1])
with left:
    if st.button("← Back to glossary"):
        st.query_params.clear()
        st.experimental_rerun()
with right:
    new_sid = st.text_input("Jump to another strain ID", value="", placeholder="Exact strain ID…").strip()
    if new_sid:
        if (df["strain_id"] == new_sid).any():
            goto_strain(new_sid)
            st.experimental_rerun()
        else:
            st.error(f"Strain '{new_sid}' not found.")

