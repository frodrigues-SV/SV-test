#!/usr/bin/env python
# coding: utf-8

# # GUI Strain Profile - 2

# In[ ]:


# gui_strain_profile_oil.py
# Single-sample viewer: pick a sample (row) and get a dedicated "page" for it.

import io
import re
from urllib.parse import urlencode  # reserved for future share links

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Strain Vaults · Oil predictions", layout="wide")
st.title("Oil Task · Single-sample viewer")

# ---------- Robust loader & normaliser (handles your CSV example) ----------
def read_any_table(uploaded) -> pd.DataFrame:
    """Auto-detect delimiter and read CSV/TSV gracefully."""
    if uploaded is None:
        return None
    b = uploaded.getvalue()
    try:
        return pd.read_csv(io.BytesIO(b), sep=None, engine="python", skipinitialspace=True)
    except Exception:
        return pd.read_csv(io.BytesIO(b))  # fallback

def looks_like_monotonic_index(col: pd.Series) -> bool:
    """True if column looks like 0..N-1 or 1..N monotonic index."""
    try:
        vals = col.astype(int).to_numpy()
    except Exception:
        return False
    n = len(vals)
    return np.array_equal(vals, np.arange(vals.min(), vals.min() + n))

def norm_col(s: str) -> str:
    """Normalise column name to a lowercase snake-ish token."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def normalise(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts files with columns like 'Alkanes', 'Aromatics', 'Biosurfactants'
    (or p_* variants), optional leading index column, optional metadata.
    Returns canonical columns:
      record_id, p_A_Alkanes, p_C_Aromatics, p_D_Biosurfactants,
      top_class, top_score, margin, entropy, [meta...]
    """
    df = df_in.copy()

    # Drop junk/empty index columns (e.g. 'Unnamed: 0')
    junk = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if junk:
        df.drop(columns=junk, inplace=True)

    # If first col is 0..N or 1..N, drop it (your pasted file case)
    if df.shape[1] >= 2 and looks_like_monotonic_index(df.iloc[:, 0]):
        df.drop(columns=[df.columns[0]], inplace=True)

    # Map prob columns from flexible names to canonical
    normmap = {c: norm_col(c) for c in df.columns}
    inv = {}
    for raw, n in normmap.items():
        inv.setdefault(n, []).append(raw)

    def find_one(*candidates):
        for c in candidates:
            if c in inv:
                return inv[c][0]
        return None

    col_alk = find_one("p_a_alkanes", "alkanes", "p_alkanes", "a_alkanes")
    col_aro = find_one("p_c_aromatics", "aromatics", "p_aromatics", "c_aromatics")
    col_bio = find_one("p_d_biosurfactants", "biosurfactants", "p_biosurfactants", "d_biosurfactants")

    if not (col_alk and col_aro and col_bio):
        raise ValueError("Could not detect all probability columns (Alkanes/Aromatics/Biosurfactants).")

    # Choose record_id if present; else synthesise from row order
    rec = None
    for cand in ["record_id", "strain_id", "id", "sample_id"]:
        if cand in df.columns:
            rec = df[cand].astype(str)
            break
    if rec is None:
        rec = pd.Series(np.arange(len(df)).astype(str), name="record_id")

    out = pd.DataFrame({"record_id": rec})
    out["p_A_Alkanes"] = pd.to_numeric(df[col_alk].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out["p_C_Aromatics"] = pd.to_numeric(df[col_aro].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out["p_D_Biosurfactants"] = pd.to_numeric(df[col_bio].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # carry through common metadata if present
    for meta in ["model_version", "preprocess_version", "inference_time", "project", "genus", "species"]:
        if meta in df.columns:
            out[meta] = df[meta].astype(str)

    # Derived columns
    P = out[["p_A_Alkanes","p_C_Aromatics","p_D_Biosurfactants"]].to_numpy(dtype=float)
    top_idx = P.argmax(axis=1)
    names = np.array(["A_Alkanes","C_Aromatics","D_Biosurfactants"])
    out["top_class"] = names[top_idx]
    out["top_score"] = P.max(axis=1)

    srt = np.sort(P, axis=1)[:, ::-1]
    out["margin"] = srt[:,0] - np.where(srt.shape[1] > 1, srt[:,1], 0.0)

    def entropy_row(x):
        s = x.sum()
        if s <= 0:
            return 0.0
        q = np.clip(x / s, 1e-12, 1.0)
        return float(-(q * np.log(q)).sum())
    out["entropy"] = np.apply_along_axis(entropy_row, 1, P)

    return out

# ---------- Upload + parse ----------
default_path = "GUI/test/test_probas.csv"
uploaded = st.file_uploader("Upload predictions (CSV/TSV)", type=["csv", "tsv", "txt"])

if uploaded is not None:
    raw = read_any_table(uploaded)
else:
    st.info("No file uploaded — using built-in demo data.")
    raw = pd.read_csv(default_path)

try:
    df = normalise(raw)
except Exception as e:
    st.error(f"Parse error: {e}")
    st.write("First rows for debugging:")
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
    st.stop()

try:
    df = normalise(raw)
except Exception as e:
    st.error(f"Parse error: {e}")
    st.write("First rows for debugging:")
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
    st.stop()

# ---------- Sidebar navigation (dropdown + next/prev) ----------
st.sidebar.header("Sample")

# Shareable URL param ?id=<record_id>
qs = st.query_params
rid_default = qs.get("id", [None])[0] if isinstance(qs.get("id"), list) else qs.get("id")

options = df["record_id"].astype(str).tolist()
default_index = options.index(rid_default) if rid_default in options else 0
sel = st.sidebar.selectbox("record_id", options, index=default_index)
idx = options.index(sel)

col_prev, col_next = st.sidebar.columns(2)
if col_prev.button("◀ Prev", use_container_width=True) and idx > 0:
    idx -= 1
if col_next.button("Next ▶", use_container_width=True) and idx < len(options) - 1:
    idx += 1
sel = options[idx]

# Update URL query param for shareable link
st.query_params.from_dict({"id": sel})

# ---------- Per-sample "page" ----------
row = df[df["record_id"] == sel].iloc[0]

st.subheader(f"Sample: {sel}")
meta_cols = [c for c in ["project","genus","species","model_version","preprocess_version","inference_time"] if c in df.columns]
if meta_cols:
    with st.expander("Metadata", expanded=True):
        meta = row[meta_cols].to_dict()
        for k, v in meta.items():
            st.write(f"**{k}**: {v}")

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Top class", str(row["top_class"]))
c2.metric("Top score", f"{row['top_score']:.3f}")
c3.metric("Margin", f"{row['margin']:.3f}")
c4.metric("Entropy", f"{row['entropy']:.3f}")

# Tabs within the sample "page"
tab_overview, tab_raw, tab_export = st.tabs(["Overview", "Raw probabilities", "Export"])

with tab_overview:
    st.write("Class probabilities")
    st.progress(min(max(float(row["p_A_Alkanes"]), 0.0), 1.0), text=f"Alkanes: {row['p_A_Alkanes']:.3f}")
    st.progress(min(max(float(row["p_C_Aromatics"]), 0.0), 1.0), text=f"Aromatics: {row['p_C_Aromatics']:.3f}")
    st.progress(min(max(float(row["p_D_Biosurfactants"]), 0.0), 1.0), text=f"Biosurfactants: {row['p_D_Biosurfactants']:.3f}")

with tab_raw:
    st.write(pd.DataFrame({
        "class": ["A_Alkanes","C_Aromatics","D_Biosurfactants"],
        "probability": [row["p_A_Alkanes"], row["p_C_Aromatics"], row["p_D_Biosurfactants"]]
    }))

with tab_export:
    st.caption("Download this sample’s row as CSV")
    out_row = df[df["record_id"] == sel]
    st.download_button(
        "Download sample CSV",
        data=out_row.to_csv(index=False).encode(),
        file_name=f"sample_{sel}.csv",
        mime="text/csv"
    )

# Optional: collapsible table of all rows
with st.expander("All predictions (table)"):
    st.dataframe(
        df[[
            "record_id","top_class","top_score","margin","entropy",
            "p_A_Alkanes","p_C_Aromatics","p_D_Biosurfactants"
        ]],
        use_container_width=True, hide_index=True
    )


# In[ ]:




