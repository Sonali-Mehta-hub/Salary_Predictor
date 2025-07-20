# -----------------------------
# Salary Prediction ML Project
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Handle imbalance
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import warnings
warnings.filterwarnings("ignore")

# --------------------------------
# Step 1: Load & Inspect the Data
# --------------------------------
print("\n🔹 Loading dataset...")
df = pd.read_excel(
    r"C:\Users\AROBASE\Desktop\Major Projects\SalaryPredictor\Salary_Predictor\dataset\adult 3.xlsx"
)
print("✅ Data loaded successfully.")
print(f"Initial data shape: {df.shape}")
print(df.head())

# --------------------------------
# Step 2: Clean the Data
# --------------------------------
print("\n🔹 Cleaning data...")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Drop redundant education column if both exist
if 'education' in df.columns and 'education-num' in df.columns:
    df.drop('education', axis=1, inplace=True)

# Remove outlier ages
df = df[(df['age'] >= 17) & (df['age'] <= 75)]

# Encode the target variable
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

print(f"✅ Cleaned data shape: {df.shape}")
print(df['income'].value_counts())

# --------------------------------
# Step 3: Feature Engineering
# --------------------------------
X = df.drop('income', axis=1)
y = df['income']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

print("\n🔹 Features separated.")
print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)

# ✅ FIXED: handle_unknown='ignore' prevents crashing on unseen categories
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# --------------------------------
# Step 4: Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🔹 Splitting data...")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Before SMOTE:\n", y_train.value_counts())

# Apply preprocessing
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
print("✅ Data preprocessing complete.")

# --------------------------------
# Step 5: Balance the Dataset with SMOTE
# --------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_encoded, y_train)

print("\n🔹 After applying SMOTE:")
print(y_train_res.value_counts())

# --------------------------------
# Step 6: Define & Train Models
# --------------------------------
print("\n🔹 Training models and evaluating performance...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"👉 Training {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_encoded)
    y_proba = model.predict_proba(X_test_encoded)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0

    results[name] = {"Accuracy": acc, "F1-score": f1, "ROC-AUC": roc}

    print(f"✅ {name} → Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------------")

# --------------------------------
# Step 7: Compare Models
# --------------------------------
results_df = pd.DataFrame(results).T
print("\n📊 Final Model Comparison:")
print(results_df)

# Sort by ROC-AUC before visualizing
results_df = results_df.sort_values(by="ROC-AUC", ascending=False)

# Bar chart
results_df[['Accuracy', 'F1-score', 'ROC-AUC']].plot(kind='bar', figsize=(12,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------
# Step 8: Save Best Model + Preprocessor
# --------------------------------
best_model_name = results_df['ROC-AUC'].idxmax()
best_model = models[best_model_name]
best_model.fit(X_train_res, y_train_res)

# ✅ Save the best model
joblib.dump(best_model, "best_salary_model.pkl")

# ✅ Save the fitted preprocessor so Streamlit can use it
joblib.dump(preprocessor, "preprocessor.pkl")

print(f"\n🏆 Best Model: {best_model_name}")
print("✅ Saved as 'best_salary_model.pkl' & 'preprocessor.pkl'")
