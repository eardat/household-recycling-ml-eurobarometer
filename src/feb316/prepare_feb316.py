from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FEB316 = ROOT / "data" / "raw" / "ZA5474_FEB316" / "ZA5474_v1-0-0.xlsx"
PROCESSED_FEB316 = ROOT / "data" / "processed" / "FEB316"
RANDOM_STATE = 42

NUMERIC_BASE = ["d3_edu", "gender", "age_group", "area_type"]
CAT_COLS = ["activity", "occupation_group"]


def load_raw(raw_path=RAW_FEB316):
    return pd.read_excel(raw_path)


def processed_paths():
    return {
        "X_train": PROCESSED_FEB316 / "X_train.csv",
        "X_test": PROCESSED_FEB316 / "X_test.csv",
        "y_train": PROCESSED_FEB316 / "y_train.csv",
        "y_test": PROCESSED_FEB316 / "y_test.csv",
    }


def build_features(raw_path=RAW_FEB316):
    df_raw = pd.read_excel(raw_path)
    feature_cols = ["d1", "d2_r", "d3_r", "d4", "activity", "d6", "country"]
    target_col = "q2"
    df = df_raw[feature_cols + [target_col]].copy()
    df = df.replace(["DK/NA", "Refusal", "refusal"], np.nan)
    df = df.dropna(subset=[target_col])

    occ_map = {
        "general management, director or top management": "manager",
        "manager of a company": "manager",
        "middle management": "manager",
        "professional (employed doctor, lawyer, accountant, architect": "professional",
        "professional (lawyer, medical practitioner, accountant, arch": "professional",
        "Civil servant": "professional",
        "supervisor / foreman (team manager, etc...)": "professional",
        "owner of a shop, craftsman": "self_employed",
        "other employee (salesman, nurse, etc...)": "employee",
        "office clerk": "employee",
        "Manual worker": "manual_worker",
        "unskilled manual worker": "manual_worker",
        "farmer, forester, fisherman": "manual_worker",
        "retired": "retired",
        "looking after the home": "not_working",
        "seeking a job": "not_working",
        "student (full time)": "student",
        "other": "not_working",
    }
    edu_map = {
        " never": 0.0,
        " -15": 1.0,
        "16-20": 2.0,
        "20+": 3.0,
        "still in education": 3.0,
    }
    age_map = {"15-24": 0.0, "25-39": 1.0, "40-54": 2.0, "55+": 3.0}
    area_map = {"Rural zone": 0.0, "Other town/urban centre": 1.0, "Metropolitan zone": 2.0}

    X = pd.DataFrame(index=df.index)
    X["d3_edu"] = df["d3_r"].map(edu_map)
    X["gender"] = df["d1"].map({"Female": 0.0, "Male": 1.0})
    X["age_group"] = df["d2_r"].map(age_map)
    X["area_type"] = df["d6"].map(area_map)
    X["activity"] = df["activity"].astype("object")
    X["occupation_group"] = df["d4"].map(occ_map).astype("object")
    X["country_iso"] = df["country"].astype("object")

    y = df[target_col].map({"No": 0, "Yes": 1})
    mask = y.notna()
    return X.loc[mask], y.loc[mask].astype(int)
