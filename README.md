# DATA Analysis Project: Binary Classification
## Predicting Income Level Using the Adult Dataset

---

### **Project Identification Card:**
- **Student:** Fares Aouadi
- **Academic Level:** First-Year Master's in Artificial Intelligence (Master 1 AI)
- **Institution:** University of El Oued (Algeria)
- **Group:** 2

---

### **Project Abstract:**
This applied project aims to build and deploy a specialized machine learning model for **Binary Classification** to predict whether an individual's income exceeds $50K annually, using the benchmark `adult.csv` dataset. The workflow encompasses Exploratory Data Analysis (EDA), data cleaning and handling, categorical variable encoding, model training, systematic model comparison, and hyperparameter tuning.

---

### **Summary of Tools, Libraries, and Expected Outputs:**

| Phase | Key Libraries & Functions | Expected Output |
| :--- | :--- | :--- |
| **1. Exploration & Cleaning** | `pandas`, `numpy`, `SimpleImputer` | Clean dataset free from missing values, trailing spaces, and redundant header rows. |
| **2. Encoding & Scaling** | `LabelEncoder`, `StandardScaler`, `pd.get_dummies` | Standardized numerical feature matrix ($X$) and binary encoded target vector ($y$). |
| **3. Training & Tuning** | `train_test_split`, `XGBoost`, `RandomizedSearchCV`, `cv=5` | Optimized, cross-validated ensemble model with fine-tuned hyperparameters. |
| **4. Evaluation & Persistence** | `confusion_matrix`, `classification_report`, `joblib` | Comprehensive performance metrics, systematic comparison, and saved model artifacts (`.pkl`). |

---

### **Summary of Experiments & Results:**
1. **Baseline Model (`LogisticRegression`):** Accuracy: `81.42%` | F1-Score: `0.5085`.
2. **Advanced Ensemble Models:** 
   - **Random Forest:** F1-Score: `0.5653`.
   - **XGBoost & LightGBM:** **XGBoost** emerged as the top performer (Accuracy: `84.59%`, F1-Score: `0.6026`).
3. **Hyperparameter Optimization (`Tuned XGBoost`):** Fine-tuned via `RandomizedSearchCV` and 5-fold cross-validation (`cv=5`), raising the final Accuracy to `84.78%` and F1-Score to `60.53%`.

---

### **Project Structure:**
- `project_data_analysis.ipynb`: The complete step-by-step Jupyter notebook.
- `best_xgb_model.pkl`: Serialized and optimized XGBoost classifier.
- `scaler.pkl`: Fitted StandardScaler pipeline.
- `label_encoder.pkl`: Target label encoder artifact.
