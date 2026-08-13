import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("model.joblib")


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Student Package Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .prediction-label {
        font-size: 18px;
        color: #666;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
        margin-top: 10px;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.06);
        text-align: center;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 CGPA to Package Predictor of Students</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based salary prediction system'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎯 Enter Details")

    cgpa = st.number_input(
        "Enter your CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    st.write("")

    predict_button = st.button(
        "🚀 Predict Package"
    )

    st.divider()

    st.info(
        "Enter a CGPA between 0 and 10. "
        "The trained ML model will predict the expected package."
    )


# =========================================================
# MAIN CONTENT
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="info-card">'
        '<h3>📚 Input</h3>'
        '<h2>CGPA</h2>'
        '<p>Academic Score</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="info-card">'
        '<h3>🤖 Model</h3>'
        '<h2>ML Model</h2>'
        '<p>Regression</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="info-card">'
        '<h3>💼 Output</h3>'
        '<h2>Package</h2>'
        '<p>Expected LPA</p>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Prepare input
    input_data = np.array([[cgpa]])

    # Prediction
    prediction = model.predict(input_data)

    # Convert numpy array to scalar
    predicted_package = float(
        np.asarray(prediction).flatten()[0]
    )

    st.markdown("## 📊 Prediction Result")

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Predicted Package
            </div>

            <div class="prediction-value">
                ₹ {predicted_package:.2f} LPA
            </div>

            <p>
                Based on CGPA: <b>{cgpa:.2f}</b>
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # =====================================================
    # MODEL PREDICTION GRAPH
    # =====================================================

    st.subheader("📈 CGPA vs Predicted Package")

    # Generate CGPA values
    cgpa_values = np.linspace(0, 10, 50)

    # Create input for model
    cgpa_input = cgpa_values.reshape(-1, 1)

    # Predict package for each CGPA
    package_predictions = model.predict(cgpa_input)

    # Convert prediction to 1D
    package_predictions = np.asarray(
        package_predictions
    ).flatten()

    # Create dataframe
    graph_data = pd.DataFrame({
        "CGPA": cgpa_values,
        "Package": package_predictions
    })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        graph_data["CGPA"],
        graph_data["Package"],
        linewidth=3
    )

    # Highlight user's prediction
    ax.scatter(
        cgpa,
        predicted_package,
        s=150,
        zorder=5
    )

    ax.annotate(
        f"Your Prediction\n₹ {predicted_package:.2f} LPA",
        (cgpa, predicted_package),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=10
    )

    ax.set_title(
        "CGPA vs Predicted Package",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("CGPA")
    ax.set_ylabel("Package (LPA)")

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


    # =====================================================
    # PREDICTION RANGE
    # =====================================================

    st.subheader("📊 Package Prediction Details")

    min_package = float(np.min(package_predictions))
    max_package = float(np.max(package_predictions))

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Your CGPA",
            f"{cgpa:.2f}"
        )

    with c2:
        st.metric(
            "Predicted Package",
            f"₹ {predicted_package:.2f} LPA"
        )

    with c3:
        st.metric(
            "Model Package Range",
            f"₹ {min_package:.2f} - ₹ {max_package:.2f} LPA"
        )


    # =====================================================
    # DATA TABLE
    # =====================================================

    with st.expander("🔍 View Prediction Data"):

        display_data = graph_data.copy()

        display_data["CGPA"] = display_data["CGPA"].round(2)
        display_data["Package"] = display_data["Package"].round(2)

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

else:

    st.markdown("## 👈 Enter your CGPA")

    st.info(
        "Enter your CGPA in the sidebar and click "
        "**Predict Package** to see the prediction and graphs."
    )