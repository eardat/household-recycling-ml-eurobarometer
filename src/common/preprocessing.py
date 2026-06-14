import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CountryTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="country_iso", output_column="country_recycle_rate", drop_original=True):
        self.column = column
        self.output_column = output_column
        self.drop_original = drop_original

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        y_series = pd.Series(y, index=X_df.index, name="target")
        tmp = pd.DataFrame({self.column: X_df[self.column], "target": y_series})
        self.global_mean_ = float(y_series.mean())
        self.mapping_ = tmp.groupby(self.column)["target"].mean().to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df[self.output_column] = X_df[self.column].map(self.mapping_).fillna(self.global_mean_).astype(float)
        if self.drop_original:
            X_df = X_df.drop(columns=[self.column])
        return X_df


def edu_recode(age):
    if pd.isna(age):
        return np.nan
    if age <= 15:
        return 1.0
    if age <= 20:
        return 2.0
    return 3.0


def make_preprocessor(numeric_cols, cat_cols, scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric_cols),
        ("categorical", cat_pipe, cat_cols),
    ], sparse_threshold=0)


def clean_feature_name(name):
    name = str(name)
    for prefix in ["numeric__", "categorical__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.startswith("occupation_group_"):
        name = "occupation_group=" + name[len("occupation_group_"):]
    return name
