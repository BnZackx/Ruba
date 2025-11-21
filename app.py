"""
Final production-ready app.py
Features:
- Bilingual UI (English / Hausa)
- Comparison mode (A vs B)
- Constant & stepwise temperature profiles (editable)
- Uncertainty bands and optional 95% CI
- Retention classification with colored badges
- Save/load presets (JSON) and manage presets UI
- Add/save user-defined crops (JSON)
- Per-segment breakdown and CSV download
- Export multi-page PDF report (uses matplotlib.backends.backend_pdf.PdfPages)
- Download simulated time-series CSV
Author: Umar Faruk Zakariyyya | BnZackx
Date: 2025
"""
import os
import json
import math
from typing import Dict, Any, List, Tuple, Optional
import io

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------
# Constants & storage file names
# ------------------------
R_GAS = 8.314  # J/(mol*K)
MODEL_PICKLE = "vitamin_c_predictor.pkl"
USER_CROPS_FILE = "user_crops.json"        # persistent while app instance lives
PRESETS_FILE = "presets.json"
DEFAULT_KINETIC_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "Orange (Citrus sinensis)": {"C0": 52.3, "Ea": 68.4 * 1000, "A": 2.34e10},
    "Baobab (Adansonia digitata)": {"C0": 225.8, "Ea": 72.9 * 1000, "A": 1.07e11},
    "Fluted pumpkin (Telfairia occidentalis)": {"C0": 85.2, "Ea": 66.8 * 1000, "A": 8.45e9},
    "Spinach (Amaranthus hybridus)": {"C0": 62.4, "Ea": 75.2 * 1000, "A": 3.89e11},
}

# ------------------------
# Strings (en / ha)
# ------------------------
STRINGS = {
    "en": {
        "title": "BnZackx — Vitamin C Degradation Predictor",
        "subtitle": "Dept. of Food Science & Technology, ADUSTECH",
        "model_config": "Model configuration",
        "upload_prompt": "Upload kinetics JSON/CSV (optional)",
        "load_pickle": "Load predictor from pickle (.pkl)",
        "prediction_panel": "Prediction panel",
        "select_crop": "Select crop",
        "add_crop": "Add a new crop",
        "crop_name": "Crop name",
        "C0_label": "C₀ (mg/100g)",
        "Ea_label": "Ea (kJ/mol)",
        "A_label": "A (s⁻¹)",
        "presets": "Presets",
        "save_preset": "Save current as preset",
        "preset_name": "Preset name",
        "profile_type": "Temperature profile type",
        "constant": "Constant temperature",
        "stepwise": "Step-wise profile (editable)",
        "upload_profile": "Upload profile CSV (duration_min,temp_C)",
        "edit_profile": "Edit profile below",
        "compare_mode": "Enable comparison mode (A vs B)",
        "calculate": "Calculate & Plot",
        "export_pdf": "Export PDF report",
        "download_csv": "Download CSV (time-series)",
        "per_segment": "Per-segment breakdown",
        "retention": "Retention",
        "initial": "Initial (C₀)",
        "final": "Final (Cₜ)",
        "k_val": "k",
        "half_life": "Half-life (t½)",
        "uncertainty": "Uncertainty ±%",
        "show_95": "Show approx. 95% CI",
        "history": "History (session)",
        "export_history": "Export history CSV",
        "save_crop": "Save crop to user list",
        "delete_preset": "Delete preset",
        "notes": "Notes: Ea can be kJ/mol in CSV (auto converted). A assumed in s⁻¹.",
        "excellent": "Excellent (≥80%)",
        "moderate": "Moderate (50–80%)",
        "poor": "Poor (25–50%)",
        "severe": "Severe (<25%)",
        "no_predictions": "No predictions yet.",
    },
    "ha": {
        "title": "BnZackx — Ma'aunin Raguwa Vitamin C",
        "subtitle": "Sashen Kimiyyar Abinci, ADUSTECH",
        "model_config": "Saitin Samfuri",
        "upload_prompt": "Loda kinetics JSON/CSV (na zaɓi)",
        "load_pickle": "Loda predictor daga pickle (.pkl)",
        "prediction_panel": "Allon Hasashe",
        "select_crop": "Zaɓi amfanin gona",
        "add_crop": "Ƙara sabon amfanin gona",
        "crop_name": "Sunan amfanin gona",
        "C0_label": "C₀ (mg/100g)",
        "Ea_label": "Ea (kJ/mol)",
        "A_label": "A (s⁻¹)",
        "presets": "Saituna",
        "save_preset": "Ajiye a matsayin saitin",
        "preset_name": "Sunan saitin",
        "profile_type": "Nau'in bayanin zafin jiki",
        "constant": "Dindindin",
        "stepwise": "Matakai (gyarawa)",
        "upload_profile": "Loda CSV na profile (duration_min,temp_C)",
        "edit_profile": "Gyara profile a ƙasa",
        "compare_mode": "Banda kwatantawa (A vs B)",
        "calculate": "Lissafi & Zanewa",
        "export_pdf": "Fitar da rahoton PDF",
        "download_csv": "Zazzage CSV (lokaci-series)",
        "per_segment": "Bayanin kowane mataki",
        "retention": "Adana",
        "initial": "Farko (C₀)",
        "final": "Na ƙarshe (Cₜ)",
        "k_val": "k",
        "half_life": "Rabi-rayuwa (t½)",
        "uncertainty": "Rashin tabbas ±%",
        "show_95": "Nuna 95% CI (kimanin)",
        "history": "Tarihi (zama)",
        "export_history": "Fitar da tarihin CSV",
        "save_crop": "Ajiye amfanin gona",
        "delete_preset": "Goge saitin",
        "notes": "Lura: Ea na iya zama kJ/mol a CSV (ana sauyawa). A auna a s⁻¹.",
        "excellent": "Babban (≥80%)",
        "moderate": "Matsakaici (50–80%)",
        "poor": "Ragowa (25–50%)",
        "severe": "Mummuna (<25%)",
        "no_predictions": "Babu hasashen tukuna.",
    }
}

# ------------------------
# Helper: load & save JSON resources
# ------------------------
def load_json_or_default(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: str, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save {path}: {e}")


# ------------------------
# Kinetics model
# ------------------------
class Predictor:
    def __init__(self, params: Dict[str, Dict[str, Any]]):
        self.params = params

    def get_k(self, crop: str, temp_c: float, A_unit: str = "s^-1") -> float:
        if crop not in self.params:
            raise KeyError("Crop not found.")
        Ea = float(self.params[crop]["Ea"])
        A = float(self.params[crop]["A"])
        if temp_c <= -273.15:
            raise ValueError("Temperature invalid.")
        T_K = temp_c + 273.15
        k_s = A * math.exp(-Ea / (R_GAS * T_K))
        return k_s * 60.0 if A_unit == "s^-1" else k_s

    def simulate_profile(self, crop: str, profile: List[Tuple[float, float]],
                         C0_override: Optional[float], uncertainty_frac: float):
        # profile: list of (duration_min, temp_C)
        if crop not in self.params:
            raise KeyError("Crop not found.")
        C0 = float(self.params[crop]["C0"]) if C0_override is None else float(C0_override)
        t_acc = 0.0
        Ct = C0
        rows = []
        for dur, temp in profile:
            if dur <= 0:
                continue
            # choose number of substeps for smoothness
            n_steps = max(2, int(min(200, math.ceil(dur * 4))))
            dt = dur / n_steps
            k = self.get_k(crop, temp)
            for _ in range(n_steps):
                t_acc += dt
                Ct = Ct * math.exp(-k * dt)
                rows.append({
                    "time_min": t_acc,
                    "temp_C": temp,
                    "Ct": Ct,
                    "Ct_low": Ct * (1 - uncertainty_frac),
                    "Ct_high": Ct * (1 + uncertainty_frac),
                    "k": k
                })
        return pd.DataFrame(rows)

# ------------------------
# Utility: retention classification badge
# ------------------------
def retention_badge(pct: float, lang: str = "en") -> str:
    s = STRINGS[lang]
    if pct >= 80:
        text = s["excellent"]; color = "#2d9a2d"
    elif pct >= 50:
        text = s["moderate"]; color = "#1f77b4"
    elif pct >= 25:
        text = s["poor"]; color = "#ff9800"
    else:
        text = s["severe"]; color = "#d7263d"
    # return HTML badge
    return f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:600'>{text} · {pct:.1f}%</div>"

# ------------------------
# App start
# ------------------------
st.set_page_config(page_title="Vitamin C Predictor", layout="wide")
lang = st.sidebar.selectbox("Language / Harshe", ["en", "ha"], index=0)
S = STRINGS[lang]

st.markdown(f"""
<div style="text-align:center">
  <h2 style="margin:0;color:green">{S['title']}</h2>
  <div style="color:gray">{S['subtitle']}</div>
</div>
""", unsafe_allow_html=True)
st.write("---")

# load user crops & presets (persistent files)
user_crops = load_json_or_default(USER_CROPS_FILE, {})
presets = load_json_or_default(PRESETS_FILE, {})

# left config column
c1, c2 = st.columns([2, 1])
with c1:
    st.header(S["model_config"])
    uploaded = st.file_uploader(S["upload_prompt"], type=["json", "csv"])
    st.markdown(S["notes"])
with c2:
    use_pickle = st.checkbox(S["load_pickle"], False)
    # show saved crops management
    st.subheader(S["add_crop"])
    with st.form("add_crop_form", clear_on_submit=True):
        new_name = st.text_input(S["crop_name"], "")
        c0_in = st.number_input(S["C0_label"], min_value=0.0, value=0.0, step=0.1)
        ea_in = st.number_input(S["Ea_label"], min_value=0.0, value=50.0, step=0.1)
        a_in = st.number_input(S["A_label"], min_value=0.0, value=1e10, format="%.5e", step=1e8)
        if st.form_submit_button(S["save_crop"]):
            if not new_name:
                st.warning("Provide a crop name.")
            else:
                # store Ea in J/mol
                user_crops[new_name] = {"C0": c0_in, "Ea": ea_in * 1000.0, "A": a_in}
                save_json(USER_CROPS_FILE, user_crops)
                st.success(f"Saved crop: {new_name}")

# merge base + user crops
kinetics = {**DEFAULT_KINETIC_PARAMETERS, **user_crops}

# uploaded kinetics file overrides
if uploaded:
    try:
        if uploaded.name.endswith(".json"):
            loaded = json.load(uploaded)
            # validate
            valid = True
            for k, v in loaded.items():
                if not all(x in v for x in ("C0", "Ea", "A")):
                    valid = False; break
            if valid:
                kinetics = loaded
                st.success("Kinetics loaded from JSON.")
            else:
                st.error("Invalid kinetics JSON format. Expect dict of {crop: {C0,Ea,A}}")
        else:
            df_k = pd.read_csv(uploaded)
            # expect columns: crop, C0, Ea (kJ or J), A
            tmp = {}
            for _, r in df_k.iterrows():
                crop = str(r.iloc[0])
                C0 = float(r.iloc[1])
                Ea_raw = float(r.iloc[2])
                A_raw = float(r.iloc[3])
                Ea = Ea_raw * 1000.0 if Ea_raw < 10000 else Ea_raw
                tmp[crop] = {"C0": C0, "Ea": Ea, "A": A_raw}
            kinetics = tmp
            st.success("Kinetics loaded from CSV.")
    except Exception as e:
        st.error(f"Could not load kinetics: {e}")

predictor = Predictor(kinetics)

# ===== MAIN PANEL =====
st.header(S["prediction_panel"])
left, right = st.columns([1, 2])

with left:
    crop_a = st.selectbox(S["select_crop"], list(kinetics.keys()), index=0, key="crop_a")
    # presets UI
    st.write("**" + S["presets"] + "**")
    preset_choice = st.selectbox("Load preset", ["-- none --"] + list(presets.keys()))
    if preset_choice and preset_choice != "-- none --":
        # preset structure expected: {'profile':[ [dur,temp],... ], 'C0_override': None, 'uncertainty_pct':5}
        p = presets[preset_choice]
        # store in session for later application
        st.session_state["preset_loaded"] = p
        st.success(f"Preset '{preset_choice}' loaded (applied on Calculate).")
    # save preset
    st.text_input(S["preset_name"], key="preset_name_input")
    if st.button(S["save_preset"]):
        pname = st.session_state.get("preset_name_input", "").strip()
        if not pname:
            st.warning("Provide preset name.")
        else:
            # create current default preset placeholder (user can edit after calculate)
            presets[pname] = {"profile": [[15.0, 85.0]], "C0_override": None, "uncertainty_pct": 5.0}
            save_json(PRESETS_FILE, presets)
            st.success(f"Preset '{pname}' saved.")

    # choose profile type
    profile_type = st.radio(S["profile_type"], (S["constant"], S["stepwise"]))
    if profile_type == S["constant"]:
        temp_a = st.slider(S["profile_type"] + " — " + S["temperature"] if "temperature" in S else "Temperature", 0.0, 200.0, 85.0, 0.5, key="temp_a")
        time_a = st.number_input(S["time_min"] if "time_min" in S else "Time (min)", 0.0, 10000.0, 15.0, 0.1, key="time_a")
        profile_a = [(float(time_a), float(temp_a))]
    else:
        st.write(S["upload_profile"])
        prof_file = st.file_uploader("Profile A CSV", type=["csv"], key="profA")
        if prof_file:
            dfp = pd.read_csv(prof_file)
            profile_a = [(float(row[0]), float(row[1])) for row in dfp.values]
        else:
            if "profile_a_df" not in st.session_state:
                st.session_state["profile_a_df"] = pd.DataFrame({"duration_min": [5.0, 5.0, 5.0], "temp_C": [80.0, 90.0, 85.0]})
            st.session_state["profile_a_df"] = st.experimental_data_editor(st.session_state["profile_a_df"], num_rows="dynamic")
            profile_a = [(float(r["duration_min"]), float(r["temp_C"])) for _, r in st.session_state["profile_a_df"].iterrows()]

    compare = st.checkbox(S["compare_mode"], key="compare_box")
    if compare:
        crop_b = st.selectbox("B — " + S["select_crop"], list(kinetics.keys()), index=1, key="crop_b")
        if profile_type == S["constant"]:
            temp_b = st.slider("B — temp (°C)", 0.0, 200.0, 90.0, 0.5, key="temp_b")
            time_b = st.number_input("B — time (min)", 0.0, 10000.0, 10.0, 0.1, key="time_b")
            profile_b = [(float(time_b), float(temp_b))]
        else:
            prof_file_b = st.file_uploader("Profile B CSV", type=["csv"], key="profB")
            if prof_file_b:
                dfpb = pd.read_csv(prof_file_b)
                profile_b = [(float(row[0]), float(row[1])) for row in dfpb.values]
            else:
                if "profile_b_df" not in st.session_state:
                    st.session_state["profile_b_df"] = pd.DataFrame({"duration_min": [5.0, 5.0], "temp_C": [90.0, 95.0]})
                st.session_state["profile_b_df"] = st.experimental_data_editor(st.session_state["profile_b_df"], num_rows="dynamic")
                profile_b = [(float(r["duration_min"]), float(r["temp_C"])) for _, r in st.session_state["profile_b_df"].iterrows()]
    else:
        crop_b = None
        profile_b = None

    C0_override_a = st.number_input(S["C0_label"], min_value=0.0, value=0.0, step=0.1, key="C0_override_a")
    if C0_override_a == 0.0:
        C0_override_a = None
    uncertainty_pct = st.slider(S["uncertainty"], 0.0, 50.0, 5.0, 0.5, key="uncertainty_pct")
    uncertainty_frac = uncertainty_pct / 100.0
    show_95 = st.checkbox(S["show_95"], value=False)

    st.write("---")
    if st.button(S["calculate"]):
        try:
            # apply preset override if loaded
            if "preset_loaded" in st.session_state:
                p = st.session_state["preset_loaded"]
                profile_a = [(float(x[0]), float(x[1])) for x in p.get("profile", profile_a)]
                C0_override_a = p.get("C0_override", C0_override_a)
                uncertainty_frac = (p.get("uncertainty_pct", uncertainty_pct) / 100.0)

            df_a = predictor.simulate_profile(crop_a, profile_a, C0_override_a, uncertainty_frac)
            if df_a.empty:
                st.error("Simulation returned no data for A.")
            else:
                Ct_a = float(df_a.iloc[-1]["Ct"])
                C0_used_a = predictor.params[crop_a]["C0"] if C0_override_a is None else C0_override_a
                retention_a = (Ct_a / C0_used_a) * 100 if C0_used_a > 0 else 0.0
                # display A results
                st.subheader(f"A — {crop_a}")
                st.markdown(retention_badge(retention_a, lang), unsafe_allow_html=True)
                st.write(f"{S['final']} {Ct_a:.2f} mg/100g")
                st.write(f"{S['initial']} {C0_used_a:.2f} mg/100g")
                st.write(f"{S['k_val']}: {df_a.iloc[-1]['k']:.4e} min⁻¹")
                st.write(f"{S['half_life']}: {math.log(2) / df_a.iloc[-1]['k']:.2f} min")
                # plot A
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(df_a["time_min"], df_a["Ct"], label=f"A: {crop_a}", linewidth=2)
                if uncertainty_frac > 0:
                    ax.fill_between(df_a["time_min"], df_a["Ct_low"], df_a["Ct_high"], alpha=0.25, label=f"A ±{uncertainty_pct}%")
                    if show_95:
                        low95 = df_a["Ct"] * (1 - 1.96 * uncertainty_frac)
                        high95 = df_a["Ct"] * (1 + 1.96 * uncertainty_frac)
                        ax.plot(df_a["time_min"], low95, linestyle="--", alpha=0.6)
                        ax.plot(df_a["time_min"], high95, linestyle="--", alpha=0.6)
                # If comparison
                if compare and crop_b and profile_b:
                    df_b = predictor.simulate_profile(crop_b, profile_b, None, uncertainty_frac)
                    if not df_b.empty:
                        Ct_b = float(df_b.iloc[-1]["Ct"])
                        C0_used_b = predictor.params[crop_b]["C0"]
                        retention_b = (Ct_b / C0_used_b) * 100 if C0_used_b > 0 else 0.0
                        st.subheader(f"B — {crop_b}")
                        st.markdown(retention_badge(retention_b, lang), unsafe_allow_html=True)
                        st.write(f"{S['final']} {Ct_b:.2f} mg/100g")
                        st.write(f"{S['initial']} {C0_used_b:.2f} mg/100g")
                        ax.plot(df_b["time_min"], df_b["Ct"], label=f"B: {crop_b}", linewidth=2)
                        if uncertainty_frac > 0:
                            ax.fill_between(df_b["time_min"], df_b["Ct_low"], df_b["Ct_high"], alpha=0.15, label=f"B ±{uncertainty_pct}%")

                ax.set_xlabel("Time (min)")
                ax.set_ylabel("Vitamin C (mg/100g)")
                ax.set_title("Vitamin C degradation")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                # per-segment breakdown for A
                st.write("### " + S["per_segment"])
                # create breakdown by profile segments
                breakdown = []
                t_prev = 0.0
                Ct_prev = predictor.params[crop_a]["C0"] if C0_override_a is None else C0_override_a
                for dur, temp in profile_a:
                    kseg = predictor.get_k(crop_a, temp)
                    Ct_after = Ct_prev * math.exp(-kseg * dur)
                    loss_pct = ((Ct_prev - Ct_after) / Ct_prev) * 100 if Ct_prev > 0 else 0.0
                    breakdown.append({"duration_min": dur, "temp_C": temp, "Ct_before": Ct_prev, "Ct_after": Ct_after, "loss_pct": loss_pct})
                    Ct_prev = Ct_after
                    t_prev += dur
                df_break = pd.DataFrame(breakdown)
                st.dataframe(df_break.style.format({"duration_min":"{:.2f}","temp_C":"{:.1f}","Ct_before":"{:.3f}","Ct_after":"{:.3f}","loss_pct":"{:.2f}"}))

                # download buttons
                csv_a = df_a.to_csv(index=False)
                st.download_button("Download A time-series CSV", csv_a, file_name="A_timeseries.csv", mime="text/csv")
                if compare and crop_b:
                    csv_b = df_b.to_csv(index=False)
                    st.download_button("Download B time-series CSV", csv_b, file_name="B_timeseries.csv", mime="text/csv")
                st.download_button("Download A per-segment CSV", df_break.to_csv(index=False), file_name="A_breakdown.csv", mime="text/csv")

                # save to history
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append({"crop": crop_a, "Ct": Ct_a, "retention_pct": retention_a, "profile": profile_a})

                # offer PDF export
                buf = io.BytesIO()
                if st.button(S["export_pdf"]):
                    with PdfPages(buf) as pdf:
                        # Page 1: summary text
                        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4
                        fig1.clf()
                        txt = f"Vitamin C Degradation Report\nCrop A: {crop_a}\nFinal Ct: {Ct_a:.2f} mg/100g\nRetention: {retention_a:.2f}%\n"
                        if compare and crop_b:
                            txt += f"\nCrop B: {crop_b}\nFinal Ct: {Ct_b:.2f} mg/100g\nRetention B: {retention_b:.2f}%\n"
                        fig1.text(0.1, 0.9, txt, fontsize=12)
                        pdf.savefig(fig1)
                        plt.close(fig1)

                        # Page 2: plot image
                        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                        ax2.plot(df_a["time_min"], df_a["Ct"], label=f"A: {crop_a}", linewidth=2)
                        if compare and crop_b:
                            ax2.plot(df_b["time_min"], df_b["Ct"], label=f"B: {crop_b}", linewidth=2)
                        ax2.set_title("Vitamin C degradation")
                        ax2.set_xlabel("Time (min)")
                        ax2.set_ylabel("Vitamin C (mg/100g)")
                        ax2.legend()
                        ax2.grid(True)
                        pdf.savefig(fig2)
                        plt.close(fig2)

                        # Page 3: A table snapshot
                        fig3 = plt.figure(figsize=(8.27, 11.69))
                        fig3.clf()
                        fig3.text(0.02, 0.95, "A — time-series snapshot (last 10 rows)", fontsize=12)
                        tb = df_a.tail(10).reset_index(drop=True)
                        # render table as text
                        table_text = tb.to_string(index=False, float_format="{:.4f}".format)
                        fig3.text(0.02, 0.85, table_text, fontsize=8, family="monospace")
                        pdf.savefig(fig3)
                        plt.close(fig3)

                        # Page 4: A breakdown table
                        fig4 = plt.figure(figsize=(8.27, 11.69))
                        fig4.clf()
                        fig4.text(0.02, 0.95, "A — per-segment breakdown", fontsize=12)
                        fig4.text(0.02, 0.85, df_break.to_string(index=False, float_format="{:.4f}".format), fontsize=8, family="monospace")
                        pdf.savefig(fig4)
                        plt.close(fig4)

                    buf.seek(0)
                    st.download_button("Download PDF report", data=buf, file_name="vitamin_c_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Simulation error: {e}")

with right:
    st.subheader(S["history"])
    if "history" in st.session_state and st.session_state["history"]:
        hdf = pd.DataFrame(st.session_state["history"])
        st.dataframe(hdf)
        csv_hist = hdf.to_csv(index=False)
        st.download_button(S["export_history"], csv_hist, file_name="prediction_history.csv", mime="text/csv")
    else:
        st.write(S["no_predictions"])

    st.write("---")
    st.subheader("Presets")
    if presets:
        for name, p in presets.items():
            st.write(f"- {name}: profile={p.get('profile')}, uncertainty={p.get('uncertainty_pct')}")
            if st.button(f"{S['delete_preset']}: {name}"):
                presets.pop(name, None)
                save_json(PRESETS_FILE, presets)
                st.experimental_rerun()
    else:
        st.write("No presets saved.")

st.write("---")
st.markdown("<div style='text-align:center;color:gray;'>© Umar Faruk Zakariyyya | BnZackx® 2025</div>", unsafe_allow_html=True)
