from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from src.common.preprocessing import edu_recode

ROOT = Path(__file__).resolve().parents[2]
RAW_EB682 = ROOT / "data" / "raw" / "ZA4742_EB68_2" / "ZA4742_v4-0-1.sav"
RANDOM_STATE = 42

NUMERIC_BASE = [
    "gender", "age_group", "area_type", "d3_edu", "green_behavior_score",
    "env_importance", "eco_purchase", "household_size", "wealth_index",
]
CAT_COLS = ["occupation_group"]


def build_features(raw_path=RAW_EB682):
    raw_cols = [
        "v618", "v619", "v616", "v622", "v624", "v638", "v639", "v640",
        "v641", "v642", "v474", "v549", "v553", "v554", "v555", "v625", "v7", "v552",
    ]
    df, _ = pyreadstat.read_sav(str(raw_path), usecols=raw_cols, apply_value_formats=False)
    df["v616"] = df["v616"].replace([97.0, 98.0, 99.0], np.nan)
    df["v624"] = df["v624"].replace([8.0], np.nan)
    df["v474"] = df["v474"].replace([5.0], np.nan)
    df["v549"] = df["v549"].replace([5.0], np.nan)
    df = df.dropna(subset=["v552"])
    observed = set(pd.Series(df["v552"]).dropna().unique().tolist())
    if not observed.issubset({0.0, 1.0}):
        raise ValueError(f"Unexpected v552 values: {sorted(observed)}")
    y = df["v552"].astype(int)

    occ_map = {
        1.0: "self_employed", 2.0: "student", 3.0: "not_working", 4.0: "retired",
        5.0: "manual_worker", 6.0: "manual_worker", 7.0: "professional",
        8.0: "employee", 9.0: "employee", 10.0: "employee", 11.0: "professional",
        12.0: "manager", 13.0: "manager", 14.0: "not_working", 15.0: "manual_worker",
        16.0: "manual_worker", 17.0: "manual_worker", 18.0: "manual_worker",
    }
    X = pd.DataFrame(index=df.index)
    X["gender"] = df["v618"].map({1.0: 1.0, 2.0: 0.0})
    X["age_group"] = pd.cut(df["v619"], bins=[14, 24, 39, 54, 100], labels=[0, 1, 2, 3]).astype(float)
    X["area_type"] = df["v624"].map({1.0: 0.0, 2.0: 1.0, 3.0: 2.0})
    X["d3_edu"] = df["v616"].apply(edu_recode)
    X["occupation_group"] = df["v622"].map(occ_map).astype("object")
    X["country_iso"] = df["v7"].astype("object")
    X["green_behavior_score"] = df[["v553", "v554", "v555"]].sum(axis=1)
    X["env_importance"] = df["v474"].map({1.0: 4.0, 2.0: 3.0, 3.0: 2.0, 4.0: 1.0})
    X["eco_purchase"] = df["v549"].map({1.0: 4.0, 2.0: 3.0, 3.0: 2.0, 4.0: 1.0})
    X["household_size"] = df["v625"].clip(upper=6)
    X["wealth_index"] = df["v638"] * 1.5 + df["v639"] * 1.5 + df["v640"] + df["v641"] + df["v642"] * 0.5
    return X, y
