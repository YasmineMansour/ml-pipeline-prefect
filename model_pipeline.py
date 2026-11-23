import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def groupe_diag(code):
    if code == "Missing":
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    x = float(code)
    if int(x) == 250:
        return "Diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 800 <= x <= 999:
        return "Injury"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


def groupe_specialite(s):
    if s in [
        "Unknown",
        "InternalMedicine",
        "Emergency/Trauma",
        "Family/GeneralPractice",
    ]:
        return s
    if s.startswith("Cardiology"):
        return "Cardiology"
    if "Surg" in s:
        return "Surgery"
    if s.startswith("Orthopedics"):
        return "Orthopedics"
    return "Other"


def prepare_data(file_path="diabetic_data.csv"):
    print("Chargement des données...")
    df = pd.read_csv(file_path, na_values="?", keep_default_na=False, low_memory=False)

    # 4.1 Qualité des données
    df = df[~df["discharge_disposition_id"].isin([11, 19, 20, 21])]
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df.drop(columns=["examide", "citoglipton"])

    # 4.2 Valeurs manquantes
    df = df.drop(columns=["weight", "payer_code"])
    df["race"] = df["race"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df[["diag_1", "diag_2", "diag_3"]] = df[["diag_1", "diag_2", "diag_3"]].fillna(
        "Missing"
    )

    inconnus = {
        "admission_type_id": [5, 6, 8],
        "discharge_disposition_id": [18, 25, 26],
        "admission_source_id": [9, 15, 17, 20, 21],
    }
    for col, codes in inconnus.items():
        df[col] = df[col].replace(codes, 0)

    # 4.3 Regroupement des catégories
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(groupe_diag)

    df["medical_specialty"] = df["medical_specialty"].apply(groupe_specialite)

    admission_type_map = {1: "Emergency", 2: "Urgent", 3: "Elective", 0: "Unknown"}
    admission_source_map = {
        7: "Emergency",
        1: "Referral",
        2: "Referral",
        3: "Referral",
        4: "Transfer",
        5: "Transfer",
        6: "Transfer",
        10: "Transfer",
        18: "Transfer",
        22: "Transfer",
        25: "Transfer",
        26: "Transfer",
        0: "Unknown",
    }
    discharge_map = {
        1: "Home",
        6: "Home_health",
        8: "Home_health",
        3: "SNF",
        22: "Rehab",
        13: "Hospice",
        14: "Hospice",
        7: "Left_AMA",
        0: "Unknown",
    }

    df["admission_type"] = (
        df["admission_type_id"].map(admission_type_map).fillna("Unknown")
    )
    df["admission_source"] = (
        df["admission_source_id"].map(admission_source_map).fillna("Unknown")
    )
    df["discharge_disposition"] = (
        df["discharge_disposition_id"].map(discharge_map).fillna("Other_transfer")
    )

    df = df.drop(
        columns=["admission_type_id", "admission_source_id", "discharge_disposition_id"]
    )

    # 4.4 Feature engineering
    df["deja_hospitalise"] = (df["number_inpatient"] > 0).astype(int)
    df["total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    meds = list(df.loc[:, "metformin":"metformin-pioglitazone"].columns)
    df["nb_med_changes"] = df[meds].isin(["Up", "Down"]).sum(axis=1)
    df["nb_meds_diabete"] = (df[meds] != "No").sum(axis=1)

    meds_garder = [
        "insulin",
        "metformin",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "glimepiride",
    ]
    df = df.drop(columns=[m for m in meds if m not in meds_garder])

    age_map = {f"[{10*i}-{10*(i+1)})": 10 * i + 5 for i in range(10)}
    df["age"] = df["age"].map(age_map)

    # Cible
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # 4.5 Transformations
    df["number_diagnoses"] = df["number_diagnoses"].clip(upper=9)
    for col in [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "total_visits",
    ]:
        df[col] = np.log1p(df[col])

    # Variables du notebook
    num_features = [
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "total_visits",
        "nb_med_changes",
        "nb_meds_diabete",
    ]

    bin_features = ["deja_hospitalise"]

    cat_features = [
        "race",
        "gender",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "max_glu_serum",
        "A1Cresult",
        "metformin",
        "glimepiride",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "insulin",
        "change",
        "diabetesMed",
        "admission_type",
        "admission_source",
        "discharge_disposition",
    ]

    # Séparation par patient : éviter la fuite de données
    X = df.drop(columns=["readmit_30", "encounter_id", "patient_nbr"])
    y = df["readmit_30"]
    groups = df["patient_nbr"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(cv.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                cat_features,
            ),
            ("bin", "passthrough", bin_features),
        ]
    )

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "g_train": g_train,
            "preprocessor": preprocessor,
        },
        "outputs/prepared_data.joblib",
    )

    return X_train, X_test, y_train, y_test, preprocessor


def train_model():
    data = joblib.load("outputs/prepared_data.joblib")

    model = Pipeline(
        [
            ("prep", data["preprocessor"]),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Entraînement du modèle...")
    model.fit(data["X_train"], data["y_train"])

    joblib.dump(model, "outputs/model.joblib")
    print("Modèle entraîné : RandomForestClassifier")
    return model


def evaluate_model():
    data = joblib.load("outputs/prepared_data.joblib")
    model = joblib.load("outputs/model.joblib")

    y_test = data["y_test"]
    X_test = data["X_test"]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nÉvaluation du modèle :")
    print("Accuracy  :", round(accuracy_score(y_test, predictions), 3))
    print("Recall    :", round(recall_score(y_test, predictions), 3))
    print(
        "Precision :", round(precision_score(y_test, predictions, zero_division=0), 3)
    )
    print("F1-score  :", round(f1_score(y_test, predictions), 3))
    print("ROC-AUC   :", round(roc_auc_score(y_test, probabilities), 3))
    print("PR-AUC    :", round(average_precision_score(y_test, probabilities), 3))
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, predictions))


def load_model():
    model = joblib.load("outputs/model.joblib")
    print("Modèle chargé avec succès.")
    return model
