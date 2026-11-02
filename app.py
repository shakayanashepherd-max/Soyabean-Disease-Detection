
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from io import BytesIO

from auth import init_auth, register, login, current_user, logout
from database import initialize_database, save_scan, get_user_scans
from detector import load_disease_model, predict_disease


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LeafLensAI | Soybean Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DATABASE INITIALIZATION
# =========================================================

init_auth()
initialize_database()


# =========================================================
# 3. PROFESSIONAL CSS DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f8f3;
    color: #20372b;
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #174c32;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123b2a, #1d573b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1ebe1;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(25,70,40,0.06);
}

div[data-testid="stMetricLabel"] {
    color: #56715e;
}

div[data-testid="stMetricValue"] {
    color: #174c32;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #17633d;
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #10472c;
    color: white !important;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 8px;
    border: 1px solid #cbdccc;
    background: white;
    color: #20372b;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #8caf91;
    border-radius: 12px;
    padding: 12px;
}

.info-panel {
    background: white;
    border: 1px solid #e0e9df;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 15px;
}

.hero-banner {
    background: linear-gradient(110deg, #174c32, #3c8a53);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero-banner h2 {
    color: white !important;
}

.hero-banner p {
    color: #e5f4e8;
}

.result-panel {
    background: white;
    padding: 24px;
    border-radius: 15px;
    border: 1px solid #e1ebe1;
    text-align: center;
}

.login-background {
    text-align: center;
    color: white;
    padding: 30px 10px 15px;
}

.login-background h1 {
    color: white !important;
    font-size: 48px;
}

.login-background p {
    color: white;
    letter-spacing: 2px;
}

.stApp:has(.login-background) {
    background:
        linear-gradient(
            rgba(4,30,17,0.58),
            rgba(4,30,17,0.68)
        ),
        url("https://terra-xplore.com/cdn/shop/articles/Soybean_Farming_with_Agriculture_Drone_Solutions.webp?v=1762337526&width=1100")
        center center / cover no-repeat fixed !important;
}

.login-card {
    background: rgba(255,255,255,0.97);
    padding: 25px;
    border-radius: 16px 16px 0 0;
    box-shadow: 0 12px 35px rgba(0,0,0,0.2);
}

.login-title {
    color: #174c32;
    font-size: 27px;
    font-weight: 700;
    text-align: center;
}

.login-subtitle {
    color: #52695a;
    text-align: center;
}

.stApp:has(.login-background) [data-testid="stTabs"] {
    background: rgba(255,255,255,0.97);
    padding: 20px;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.2);
}

.stApp:has(.login-background) [data-testid="stTabs"] button {
    color: #174c32 !important;
    font-weight: 700;
}

.stApp:has(.login-background) [data-testid="stTabs"] button[aria-selected="true"] {
    color: white !important;
    background: #17633d !important;
    border-radius: 7px;
}

.stApp:has(.login-background) label {
    color: #174c32 !important;
    font-weight: 600;
}

.login-footer {
    color: white;
    text-align: center;
    margin-top: 25px;
    text-shadow: 1px 1px 4px #000;
}

.app-footer {
    text-align: center;
    color: #738579;
    padding: 25px 0 5px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_filename" not in st.session_state:
    st.session_state.last_filename = ""


# =========================================================
# 5. LOGIN AND REGISTRATION
# =========================================================

def authentication_page():

    st.markdown("""
    <div class="login-background">
        <div style="font-size:55px;">🌱</div>
        <h1>LeafLensAI</h1>
        <p>AI-POWERED SOYBEAN DISEASE DETECTION</p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.05, 1])

    with center:

        st.markdown("""
        <div class="login-card">
            <div class="login-title">Welcome to LeafLensAI</div>
            <div class="login-subtitle">
                Sign in or create an account to continue
            </div>
        </div>
        """, unsafe_allow_html=True)

        login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

        with login_tab:

            with st.form("login_form"):

                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    key="login_username"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password"
                )

                submitted = st.form_submit_button(
                    "Log In",
                    use_container_width=True
                )

                if submitted:

                    if login(username, password):
                        st.session_state.page = "Dashboard"
                        st.session_state.last_result = None
                        st.session_state.last_filename = ""
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with signup_tab:

            with st.form("signup_form"):

                new_username = st.text_input(
                    "Username",
                    placeholder="Choose a username",
                    key="signup_username"
                )

                new_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a password (minimum 8 characters)",
                    key="signup_password"
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Enter your password again",
                    key="signup_confirm"
                )

                submitted = st.form_submit_button(
                    "Create Account",
                    use_container_width=True
                )

                if submitted:

                    if new_password != confirm_password:
                        st.error("Passwords do not match.")

                    else:
                        success, message = register(
                            new_username,
                            new_password
                        )

                        if success:
                            st.success(message)
                            st.info("Please log in using your new account.")
                        else:
                            st.error(message)

        st.markdown("""
        <div class="login-footer">
            🌿 Smart Agriculture · Deep Learning · Crop Monitoring
            <br><br>
            © LeafLensAI | Soybean Disease Detection System
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 6. MODEL LOADING
# =========================================================

@st.cache_resource
def get_model():
    return load_disease_model()


# =========================================================
# 7. USER SCAN DATA
# =========================================================

def get_scan_dataframe(username):

    scans = get_user_scans(username)

    if not scans:
        return pd.DataFrame(
            columns=[
                "id", "username", "filename",
                "prediction", "confidence", "scan_time"
            ]
        )

    df = pd.DataFrame(scans)

    if "scan_time" in df.columns:
        df["scan_time"] = pd.to_datetime(
            df["scan_time"],
            errors="coerce"
        )

    return df


# =========================================================
# 8. DASHBOARD
# =========================================================

def dashboard_page(username):

    df = get_scan_dataframe(username)

    total_scans = len(df)
    healthy_count = 0
    rust_count = 0

    if not df.empty and "prediction" in df.columns:

        healthy_count = int(
            (df["prediction"] == "Healthy Soybean Leaf").sum()
        )

        rust_count = int(
            (df["prediction"] == "Soybean Rust").sum()
        )

    st.markdown(f"""
    <div class="hero-banner">
        <h2>Welcome back, {username} 👋</h2>
        <p>
            Monitor soybean leaf health with AI-powered disease detection.
            Upload a leaf image to begin a new analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Scans", total_scans)
    col2.metric("Healthy Leaves", healthy_count)
    col3.metric("Rust Detections", rust_count)

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("### Recent Detection Activity")

        if not df.empty:

            recent = df.head(5).copy()

            display_columns = [
                c for c in [
                    "filename", "prediction",
                    "confidence", "scan_time"
                ] if c in recent.columns
            ]

            st.dataframe(
                recent[display_columns],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No scans recorded yet. Start by uploading a soybean leaf image."
            )

    with right:

        st.markdown("### Detection Distribution")

        if total_scans > 0:

            chart_data = pd.DataFrame({
                "Category": ["Healthy", "Soybean Rust"],
                "Scans": [healthy_count, rust_count]
            })

            chart_data = chart_data[
                chart_data["Scans"] > 0
            ]

            fig = px.pie(
                chart_data,
                names="Category",
                values="Scans",
                hole=0.55
            )

            fig.update_layout(
                margin=dict(t=15, b=15, l=10, r=10),
                legend_title_text="Result"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("Your detection chart will appear here.")

    st.markdown("---")
    st.markdown("### Quick Actions")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("🌿 Detect Soybean Disease", use_container_width=True):
            st.session_state.page = "Detect Disease"
            st.rerun()

    with c2:
        if st.button("📋 View Detection History", use_container_width=True):
            st.session_state.page = "Detection History"
            st.rerun()


# =========================================================
# 9. DISEASE DETECTION
# =========================================================

def detection_page(username):

    st.title("🌿 Soybean Disease Detection")

    st.write(
        "Upload a soybean leaf image to identify whether "
        "the leaf is healthy or affected by soybean rust."
    )

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:

        st.markdown("### Upload Leaf Image")

        uploaded_file = st.file_uploader(
            "Choose a soybean leaf image",
            type=["jpg", "jpeg", "png"],
            key="leaf_upload"
        )

        if uploaded_file is not None:

            try:

                image_bytes = uploaded_file.getvalue()

                if not image_bytes:
                    st.error("The uploaded file is empty.")
                    return

                # Verify the uploaded image.
                try:
                    check_image = Image.open(
                        BytesIO(image_bytes)
                    )
                    check_image.verify()

                    # Reopen the verified image.
                    image = Image.open(
                        BytesIO(image_bytes)
                    ).convert("RGB")

                except Exception:
                    st.error(
                        "Unable to read this image. "
                        "Please upload a valid JPG, JPEG or PNG image."
                    )
                    st.info(
                        "Try opening the image on your computer. "
                        "If it does not open, use another image."
                    )
                    return

                st.image(
                    image,
                    caption="Uploaded Soybean Leaf",
                    use_container_width=True
                )

            except Exception as error:
                st.error(f"Image upload error: {error}")
                return

            if st.button(
                "🔍 Analyze Leaf Image",
                use_container_width=True,
                key="analyze_leaf_button"
            ):

                with st.spinner("AI model is analyzing your image..."):

                    try:

                        model = get_model()

                        prediction, confidence = predict_disease(image)

                        confidence = float(confidence)

                        st.session_state.last_result = {
                            "prediction": prediction,
                            "confidence": confidence
                        }

                        st.session_state.last_filename = uploaded_file.name

                        save_scan(
                            username,
                            uploaded_file.name,
                            prediction,
                            confidence
                        )

                        st.success("Image analysis completed successfully!")

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

    with right:

        st.markdown("### Analysis Result")

        result = st.session_state.last_result

        if result is not None:

            prediction = result["prediction"]
            confidence = result["confidence"]

            if prediction == "Soybean Rust":

                st.markdown("""
                <div class="result-panel">
                    <div style="font-size:55px;">⚠️</div>
                    <h2 style="color:#bd4a32;">
                        Soybean Rust Detected
                    </h2>
                    <p style="color:#63756a;">
                        The model classified this image as showing soybean rust.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-panel">
                    <div style="font-size:55px;">✅</div>
                    <h2 style="color:#23834c;">
                        Healthy Soybean Leaf
                    </h2>
                    <p style="color:#63756a;">
                        The model classified this image as healthy.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, confidence / 100))
            )

            st.caption(
                f"Image: {st.session_state.last_filename}"
            )

            st.warning(
                "This is an AI-based prediction, not a confirmed "
                "agricultural diagnosis. Verify uncertain results "
                "with an agricultural expert."
            )

        else:

            st.info(
                "Upload an image and select Analyze Leaf Image "
                "to view the prediction."
            )


# =========================================================
# 10. DETECTION HISTORY
# =========================================================

def history_page(username):

    st.title("📋 Detection History")

    st.write(
        "Review the soybean leaf images analyzed using your account."
    )

    df = get_scan_dataframe(username)

    if df.empty:
        st.info("No detection history is available yet.")
        return

    col1, col2 = st.columns(2)

    col1.metric("Total Records", len(df))

    if "prediction" in df.columns:
        rust_total = int(
            (df["prediction"] == "Soybean Rust").sum()
        )
    else:
        rust_total = 0

    col2.metric("Rust Detections", rust_total)

    st.markdown("---")

    display_columns = [
        c for c in [
            "filename", "prediction",
            "confidence", "scan_time"
        ] if c in df.columns
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    csv_data = df[display_columns].to_csv(index=False)

    st.download_button(
        "⬇️ Download Scan History",
        data=csv_data,
        file_name="leaflens_detection_history.csv",
        mime="text/csv"
    )


# =========================================================
# 11. METHODOLOGY
# =========================================================

def methodology_page():

    st.title("🧠 Project Methodology")

    st.write(
        "LeafLensAI uses deep learning and image processing "
        "to classify soybean leaves."
    )

    steps = [
        (
            "1. Image Upload",
            "The user uploads a soybean leaf image."
        ),
        (
            "2. Image Preprocessing",
            "The image is converted to RGB, resized to 224 × 224 pixels "
            "and preprocessed for MobileNetV2."
        ),
        (
            "3. Feature Extraction",
            "MobileNetV2 extracts important visual features."
        ),
        (
            "4. Classification",
            "The trained model classifies the leaf as healthy or soybean rust."
        ),
        (
            "5. Result Generation",
            "The application displays the predicted class and confidence."
        ),
        (
            "6. Database Storage",
            "The prediction is saved in the user's scan history."
        )
    ]

    for title, description in steps:

        st.markdown(f"""
        <div class="info-panel">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Classification Categories")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ Healthy Soybean Leaf")

    with col2:
        st.warning("⚠️ Soybean Rust")


# =========================================================
# 12. ABOUT PROJECT
# =========================================================

def about_page():

    st.title("🌱 About LeafLensAI")

    st.markdown("""
    LeafLensAI is an AI-powered soybean leaf disease detection
    application developed using Python, Streamlit and deep learning.

    The system uses a trained MobileNetV2 model to classify
    soybean leaf images into two categories:

    - **Healthy Soybean Leaf**
    - **Soybean Rust**

    ### Main Features

    - User registration and login
    - AI-based soybean leaf image classification
    - Prediction confidence display
    - Scan history storage
    - Dashboard analytics
    - Downloadable detection records

    ### Technologies Used

    - Python
    - Streamlit
    - TensorFlow / Keras
    - MobileNetV2
    - SQLite
    - Pillow
    - Pandas
    - Plotly

    ### Project Purpose

    The purpose of this project is to demonstrate how deep learning
    can support crop-health monitoring by analyzing soybean leaf images.
    """)


# =========================================================
# 13. PROFILE
# =========================================================

def profile_page(username):

    st.title("👤 My Profile")

    st.markdown("""
    <div class="info-panel">
        <h3>Account Information</h3>
        <p>Your LeafLensAI account details.</p>
    </div>
    """, unsafe_allow_html=True)

    st.text_input(
        "Username",
        value=username,
        disabled=True
    )

    df = get_scan_dataframe(username)

    st.metric(
        "Total Images Analyzed",
        len(df)
    )

    st.caption(
        "Your account and scan records are stored locally "
        "in the LeafLensAI SQLite database."
    )


# =========================================================
# 14. MAIN APPLICATION
# =========================================================

def main():

    username = current_user()

    if not username:
        authentication_page()
        return

    with st.sidebar:

        st.markdown("""
        <div style="text-align:center;padding:15px 5px;">
            <div style="font-size:45px;">🌱</div>
            <h2 style="color:white;margin:0;">LeafLensAI</h2>
            <p style="color:#d5eadb;font-size:12px;">
                SOYBEAN HEALTH MONITOR
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"### Welcome, {username}")
        st.markdown("### Navigation")

        pages = [
            "Dashboard",
            "Detect Disease",
            "Detection History",
            "Methodology",
            "About Project",
            "My Profile"
        ]

        selected_page = st.radio(
            "Select Page",
            pages,
            index=(
                pages.index(st.session_state.page)
                if st.session_state.page in pages else 0
            ),
            label_visibility="collapsed"
        )

        st.session_state.page = selected_page

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True):

            logout()

            st.session_state.page = "Dashboard"
            st.session_state.last_result = None
            st.session_state.last_filename = ""

            st.rerun()

        st.markdown("---")
        st.caption("AI-powered crop health monitoring")

    if st.session_state.page == "Dashboard":
        dashboard_page(username)

    elif st.session_state.page == "Detect Disease":
        detection_page(username)

    elif st.session_state.page == "Detection History":
        history_page(username)

    elif st.session_state.page == "Methodology":
        methodology_page()

    elif st.session_state.page == "About Project":
        about_page()

    elif st.session_state.page == "My Profile":
        profile_page(username)

    st.markdown("""
    <div class="app-footer">
        © 2026 LeafLensAI | Soybean Disease Detection System
        <br>
        Developed using Deep Learning and MobileNetV2
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 15. START APPLICATION
# =========================================================

if __name__ == "__main__":
    main()
