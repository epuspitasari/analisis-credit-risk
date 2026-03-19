# [Poin a] Import library yang dibutuhkan
from src import utils
from src import preprocessing
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib 

# [Poin b] Buat instance dari fastapi dengan nama app
app = FastAPI()

# [Poin c] Load model dan ohe object (Sesuai folder models Ety)
model = joblib.load("models/random_forest_best.pkl")
ohe_intent = joblib.load("models/ohe_loan_intent.pkl")
ohe_home = joblib.load("models/ohe_home_ownership.pkl")
ohe_grade = joblib.load("models/ohe_loan_grade.pkl")
ohe_default = joblib.load("models/ohe_default_on_file.pkl")

# [Poin d & e] Buat class Item
class Item(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.get("/")
def home():
    return {"message": "API Credit Risk Assessment Berjalan!"}

# [Poin f] Buat dekorator @app.post
@app.post("/predict")
# [Poin g] Buat fungsi prediksi
def predict_credit_risk(data: Item):
    # Mengambil data dari parameter dan dijadikan DataFrame
    data_dict = data.model_dump()
    df_input = pd.DataFrame([data_dict])

    # [Poin h.1] Lakukan preprocessing (Disesuaikan dengan Prefix Model Ety)
    df_processed = df_input.copy()
    
    # Mapping Prefix agar sesuai dengan feature_names_in_ model
    mapping = {
        "person_home_ownership": ("home_ownership", ohe_home),
        "loan_intent": ("loan_intent", ohe_intent),
        "loan_grade": ("loan_grade", ohe_grade),
        "cb_person_default_on_file": ("default_onfile", ohe_default)
    }

    for col, (pref, ohe_obj) in mapping.items():
        df_processed = preprocessing.ohe_transform(
            dataset = df_processed,
            subset = col,
            prefix = pref,
            ohe = ohe_obj
        )

    # MEMBUAT URUTAN KOLOM (Sesuai hasil print yang diinginkan)
    column_order = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN',
        'home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
        'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D',
        'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 'default_onfile_N',
        'default_onfile_Y'
    ]
    
    # Susun ulang kolom agar tidak error
    df_processed = df_processed[column_order]

    # [Poin h.2] Lakukan prediksi proba
    y_pred_proba = model.predict_proba(df_processed)[:, 1][0]

    # [Poin h.3] Tentukan kelas dengan threshold 0.32
    threshold = 0.32
    y_pred_class = 1 if y_pred_proba >= threshold else 0

    # [Poin h.4] Kembalikan hasil
    return {
        "status": "success",
        "prediction_proba": float(y_pred_proba),
        "prediction_class": int(y_pred_class),
        "threshold_used": threshold
    }