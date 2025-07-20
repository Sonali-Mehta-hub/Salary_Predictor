import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# ✅ Load Model & Preprocessor
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("best_salary_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# =========================
# ✅ Page Configuration
# =========================
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# =========================
# ✅ Sidebar Menu
# =========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=100)
st.sidebar.title("📌 Menu")

menu_choice = st.sidebar.radio(
    "Choose an option:",
    ["🔮 Salary Prediction", "🏠 About App"],  # Default to Prediction
    index=0
)

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 **Developed by:** Sonali")

# =========================
# ✅ Education Mapping
# =========================
education_mapping = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-acdm": 11,
    "Assoc-voc": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 16
}

# =========================
# ✅ Helper: Safe Transform
# =========================
def safe_transform(preprocessor, df):
    """Ensure all required columns exist + handle unknowns"""
    # Ensure required columns exist
    required_cols = [
        "age", "workclass", "fnlwgt", "education", "educational-num",
        "marital-status", "occupation", "relationship", "race", "gender",
        "hours-per-week", "native-country", "capital-gain", "capital-loss"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col in ["capital-gain", "capital-loss", "fnlwgt"] else "Unknown"

    # Replace '?' with 'Unknown'
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("?", "Unknown").fillna("Unknown")

    # Keep column order same as training
    df = df[required_cols]
    return preprocessor.transform(df)

# =========================
# ✅ MAIN AREA
# =========================
if menu_choice == "🏠 About App":
    # ABOUT SECTION
    st.title("💼 AI-Powered Employee Salary Predictor")
    st.markdown(
        """
        ### 👋 Welcome!  

        This app predicts **whether an employee earns >50K or <=50K** based on personal & work details.  

        ✅ **Trained on real salary dataset**  
        ✅ **Uses the best ML model automatically selected**  
        ✅ **Predict for one employee or upload a file for bulk predictions**  

        ---
        ### 🛠 How it Works  
        1️⃣ The dataset was cleaned, processed & balanced  
        2️⃣ Multiple models were trained (Logistic Regression, Random Forest, XGBoost, etc.)  
        3️⃣ The best model was chosen based on ROC-AUC  
        4️⃣ Now you can make predictions interactively  

        ---
        **Use the Sidebar → Select *Salary Prediction* to start predicting.**
        """
    )

else:
    # =========================
    # ✅ Salary Prediction Page
    # =========================
    st.title("🔮 SalarySense - AI Employee Salary Prediction")
    st.write("Choose between **Manual Prediction** or **Batch Prediction**")

    # TABS for Prediction Modes
    tab1, tab2 = st.tabs(["📝 Manual Prediction", "📂 Batch Prediction"])

    # --------------------------------
    # TAB 1: Manual Prediction
    # --------------------------------
    with tab1:
        st.subheader("📝 Predict for a Single Employee")
        st.write("Fill in the employee details below:")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 17, 75, 30)
            workclass = st.selectbox("Work Type", [
                "Private", "Self-Employed", "Government", "Never-worked", "Unknown"
            ])
            fnlwgt = st.number_input("Fnlwgt (population weight)", min_value=10000, max_value=1000000, value=200000)

            # ✅ Education dropdown
            selected_education = st.selectbox("Education Level", list(education_mapping.keys()))
            education_num = education_mapping[selected_education]

            marital_status = st.selectbox("Marital Status", [
                "Never-married", "Married", "Divorced", "Separated", "Unknown"
            ])

        with col2:
            occupation = st.selectbox("Occupation", [
                "Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Others", "Unknown"
            ])
            relationship = st.selectbox("Relationship", [
                "Husband", "Wife", "Own-child", "Not-in-family", "Unknown"
            ])
            race = st.selectbox("Race", [
                "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Unknown"
            ])
            gender = st.radio("Gender", ["Male", "Female"])
            hours_per_week = st.slider("Working Hours per Week", 1, 100, 40)
            native_country = st.selectbox("Native Country", [
                "United-States", "India", "Mexico", "Philippines", "Others", "Unknown"
            ])

        # ✅ Prepare DataFrame with all required columns
        manual_data = pd.DataFrame([{
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": selected_education,         # added education text
            "educational-num": education_num,        # mapped numeric
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "gender": gender,
            "hours-per-week": hours_per_week,
            "native-country": native_country,
            "capital-gain": 0,                       # default value
            "capital-loss": 0                        # default value
        }])

        st.write("---")
        if st.button("🔍 Predict Salary"):
            transformed = safe_transform(preprocessor, manual_data)
            prediction = model.predict(transformed)[0]
            proba = model.predict_proba(transformed)[0][1]

            if prediction == 1:
                st.success(f"💰 **Likely earning >50K!**  \n📈 Confidence: **{proba*100:.2f}%**")
            else:
                st.warning(f"📉 **Likely earning <=50K.**  \n📊 Confidence: **{(1-proba)*100:.2f}%**")

    # --------------------------------
    # TAB 2: Batch Prediction
    # --------------------------------
    with tab2:
        st.subheader("📂 Predict for Multiple Employees")
        st.write("Upload a **CSV or Excel file** with employee details.")

        uploaded_file = st.file_uploader("📤 Upload your file", type=["csv", "xlsx"])

        if uploaded_file:
            # Load uploaded file
            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.write("✅ **Uploaded File Preview:**")
            st.dataframe(batch_df.head())

            # Fill missing required columns
            for col in ["education", "capital-gain", "capital-loss"]:
                if col not in batch_df.columns:
                    batch_df[col] = "Unknown" if col == "education" else 0

            if st.button("🚀 Predict Salaries for All Employees"):
                transformed = safe_transform(preprocessor, batch_df)
                preds = model.predict(transformed)
                probs = model.predict_proba(transformed)[:, 1]

                batch_df["Predicted_Salary"] = np.where(preds == 1, ">50K", "<=50K")
                batch_df["Confidence (%)"] = (np.maximum(probs, 1 - probs) * 100).round(2)

                st.write("📊 **Prediction Results:**")
                st.dataframe(batch_df.head(10))

                # Download CSV
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    csv,
                    "salary_predictions.csv",
                    "text/csv"
                )
