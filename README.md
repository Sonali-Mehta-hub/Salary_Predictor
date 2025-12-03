# 💼 Smart Salary Prediction System

A **machine learning-powered web app** that predicts whether an employee’s salary is **>50K or <=50K** based on their demographic and work details.  
Built using **Python, Scikit-learn, XGBoost, and Streamlit**.

---

## 🚀 Features

✅ **Interactive Streamlit Web App**  
✅ **Two Modes** → Manual Prediction & Batch Prediction (CSV/Excel upload)  
✅ **Pre-trained Best ML Model**  
✅ **Confidence score for each prediction**  
✅ **Handles unknown categories & missing values gracefully**  

---

## 📂 Project Structure

📦 SalaryPredictor
 ┣ 📂 dataset
 ┃ ┗ adult.xlsx
 ┣ 📂 models
 ┃ ┣ best_salary_model.pkl
 ┃ ┗ preprocessor.pkl
 ┣ app.py
 ┣ train_model.py
 ┣ requirements.txt
 ┗ README.md



---

## ⚙️ How It Works

1️⃣ **Training Phase**  
- Loaded the **Adult Income Dataset**  
- Cleaned missing values (`?` replaced with NaN, dropped)  
- Removed outlier ages  
- Used **OneHotEncoder** for categorical features & **StandardScaler** for numerical features  
- Balanced dataset using **SMOTE**  
- Trained multiple models (Logistic Regression, Random Forest, GradientBoosting, XGBoost)  
- Selected the **best model based on ROC-AUC score**  
- Saved `best_salary_model.pkl` (model) & `preprocessor.pkl` (transformer)

2️⃣ **Prediction Phase**  
- Loads the saved model & preprocessor  
- **Manual Mode:** Fill in employee details using a clean form  
- **Batch Mode:** Upload CSV/Excel for multiple predictions  
- Outputs salary prediction with **confidence score**  

---

## 🖥️ Installation & Usage

### 1️⃣ Clone the Repository
git clone https://github.com/Sonali-Mehta-hub/Salary_Predictor
cd SalaryPredictor

2️⃣ Install Required Libraries

pip install -r requirements.txt

3️⃣ Run the Web App

streamlit run app.py
After running, the app will open in your browser (default: http://localhost:8501).

📊 Example Inputs
Manual Prediction

Age: 35
Workclass: Private
Education Years: 10
Marital Status: Married
Occupation: Sales
Hours per week: 40
Country: United States

Prediction Output →
💰 Likely earning >50K (Confidence: 87%)

Batch Prediction
Upload a CSV with the same columns → App predicts for all rows.

📷 Screenshot

## 📷 Screenshots

🏠 Home Page  
![Home Page](https://github.com/Sonali-Mehta-hub/Salary_Predictor/blob/main/assets/Screenshot%20(614).png)

🔮 Manual Prediction Form  
![Manual Prediction](https://github.com/Sonali-Mehta-hub/Salary_Predictor/blob/main/assets/Screenshot%20(612).png)

📂 Batch Prediction Results  
![Batch Prediction](https://github.com/Sonali-Mehta-hub/Salary_Predictor/blob/main/assets/Screenshot%20(615).png)



#🛠 Requirements
Here are the required libraries for this project:
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib seaborn openpyxl

✅ Quick Commands

# Train the model (if needed)
python train_model.py

# Run the Streamlit web app
 python -m streamlit run app.py



Enjoy predicting salaries with AI! 💼🤖




