import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

lift_model = joblib.load(os.path.join(BASE_DIR, "models", "svr_lift_model.pkl"))
drag_model = joblib.load(os.path.join(BASE_DIR, "models", "gradient_boosting_cd_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "lift_scaler.pkl"))

# -----------------------------
# MLflow INIT (BACKEND ONLY)
# -----------------------------
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

if MLFLOW_AVAILABLE:
    if "mlflow_init" not in st.session_state:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Airfoil_Inference_Tracking")
        st.session_state["mlflow_init"] = True

# -----------------------------
# NACA NAME
# -----------------------------
def generate_naca_name(camber, camber_pos, thickness):
    m = int(str(camber)[0])
    p = int(str(camber_pos)[0])
    t = int(thickness)
    return f"NACA {m}{p}{t:02d}"

# -----------------------------
# AOA ANALYSIS (0–10)
# -----------------------------
def aoa_sweep(camber, camber_pos, thickness):
    aoa_range = np.arange(0, 11, 1)

    cl_list, cd_list, eff_list = [], [], []

    for aoa in aoa_range:
        x = np.array([[camber, camber_pos, thickness, aoa]])
        x_scaled = scaler.transform(x)

        cl = lift_model.predict(x_scaled)[0]
        cd = drag_model.predict(x)[0]

        cl_list.append(cl)
        cd_list.append(cd)
        eff_list.append(cl / cd if cd != 0 else 0)

    best_aoa = aoa_range[np.argmax(eff_list)]

    stall_aoa = None
    for i in range(1, len(cl_list)):
        if cl_list[i] < cl_list[i - 1]:
            stall_aoa = aoa_range[i]
            break

    return aoa_range, cl_list, cd_list, eff_list, best_aoa, stall_aoa

# -----------------------------
# PDF REPORT
# -----------------------------
def generate_pdf(name, cl, cd, eff, best, stall):
    file_path = "airfoil_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Airfoil Performance Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Airfoil: {name}", styles["Normal"]),
        Paragraph(f"CL: {cl:.4f}", styles["Normal"]),
        Paragraph(f"CD: {cd:.4f}", styles["Normal"]),
        Paragraph(f"CL/CD: {eff:.4f}", styles["Normal"]),
        Paragraph(f"Best AOA: {best}", styles["Normal"]),
        Paragraph(f"Stall AOA: {stall}", styles["Normal"]),
        Paragraph("Assumption: V = 50 m/s, Wing span = 1 m", styles["Normal"]),
    ]

    doc.build(content)
    return file_path

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Airfoil MLOps System", layout="wide")

st.title("Lift and Drag prediction for NACA 4 Digit Airfoil")
st.info("Assumption: Velocity = 50 m/s | Wing span = 1 m")
st.markdown("""
## Project Overview

This Lift and Drag prediction system helps to find best geomerty of Airfoil, Lift, drag and CL/CD ratio with instant ML-based predictions.

Normally, every change in airfoil geometry requires:
- CAD redesign  
- Meshing  
- CFD simulation  
- Post-processing  

⏱ This takes **45 minutes to 1 hour per run**.

---

###  Our System Advantage

- Average prediction time: **84.37 ms**
- Traditional CFD time: **~3600 seconds**

### Time Saved
- Absolute: ~3599.92 seconds per simulation  
- Percentage: **~99.9976% faster**

---

""")
# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Input Parameters")

camber = st.sidebar.number_input("Max Camber (%)", 0.0, 9.5, 2.5)
camber_pos = st.sidebar.number_input("Camber Position (%)", 0.0, 90.0, 40.0)
thickness = st.sidebar.number_input("Thickness (%)", 1.0, 40.0, 12.0)
aoa = st.sidebar.slider("Angle of Attack", -10.0, 15.0, 5.0)

# -----------------------------
# PREDICTION
# -----------------------------
x = np.array([[camber, camber_pos, thickness, aoa]])
x_scaled = scaler.transform(x)

lift = lift_model.predict(x_scaled)[0]
drag = drag_model.predict(x)[0]
eff = lift / drag if drag != 0 else 0

name = generate_naca_name(camber, camber_pos, thickness)

# -----------------------------
# AOA ANALYSIS
# -----------------------------
aoa_range, cl_list, cd_list, eff_list, best_aoa, stall_aoa = aoa_sweep(
    camber, camber_pos, thickness
)

# -----------------------------
# TABS (REPORT MOVED TO TAB 2)
# -----------------------------
tab1, tab2 = st.tabs([
    " Prediction",
    "AOA Analysis"
])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.subheader(name)

    c1, c2, c3 = st.columns(3)
    c1.metric("CL", f"{lift:.4f}")
    c2.metric("CD", f"{drag:.4f}")
    c3.metric("CL/CD", f"{eff:.4f}")

# -----------------------------
# TAB 2 (ANALYSIS + REPORT)
# -----------------------------
with tab2:
    st.subheader("AOA Performance (0–10°)")

    fig, ax = plt.subplots()
    ax.plot(aoa_range, cl_list)
    ax.set_title("CL vs AOA")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(aoa_range, cd_list, color="red")
    ax.set_title("CD vs AOA")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(aoa_range, eff_list, color="green")
    ax.set_title("CL/CD vs AOA")
    st.pyplot(fig)

    st.success(f"Best AOA: {best_aoa}°")

    if stall_aoa:
        st.warning(f"Stall starts at ~{stall_aoa}°")

    # -----------------------------
    # REPORT SECTION (INSIDE TAB 2)
    # -----------------------------
    st.markdown("---")
    st.subheader("Generate Engineering Report")

    st.write("Download full aerodynamic report (CL, CD, CL/CD, AOA analysis)")

    if st.button("Generate PDF Report"):
        file = generate_pdf(
            name=name,
            cl=lift,
            cd=drag,
            eff=eff,
            best=best_aoa,
            stall=stall_aoa
        )

        with open(file, "rb") as f:
            st.download_button(
                "⬇ Download Report",
                f,
                file_name="airfoil_report.pdf"
            )

        st.success("Report generated successfully!")



# -----------------------------
# MLflow LOGGING (BACKEND ONLY)
# -----------------------------
st.sidebar.write("---")
st.sidebar.subheader("Experiment Logging")

if MLFLOW_AVAILABLE and st.button("📊 Log to MLflow"):
    with mlflow.start_run():

        mlflow.log_param("camber", camber)
        mlflow.log_param("camber_pos", camber_pos)
        mlflow.log_param("thickness", thickness)
        mlflow.log_param("aoa", aoa)

        mlflow.log_metric("lift", float(lift))
        mlflow.log_metric("drag", float(drag))
        mlflow.log_metric("cl_cd", float(eff))

    st.sidebar.success("Logged ✔")

