#%%
import pandas as pd
from pandas import DataFrame

from config import DATA_DIR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
           "thal", "num"]
continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ohe_columns = ["cp", "restecg", "slope", "thal", "ca"]


def get_raw_data() -> tuple[DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "processed.cleveland.data", names=columns, na_values="?")
    X = df.drop("num", axis=1)
    y = (df["num"] > 0).astype(int)
    return X, y


def process_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    # imputacja modą
    for col in ["ca", "thal"]:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # standaryzacja
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # one-hot encoding
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    train_encoded = encoder.fit_transform(X_train[ohe_columns])
    test_encoded = encoder.transform(X_test[ohe_columns])
    new_cols = encoder.get_feature_names_out(ohe_columns)

    df_train_encoded = pd.DataFrame(train_encoded, columns=new_cols, index=X_train.index)
    df_test_encoded = pd.DataFrame(test_encoded, columns=new_cols, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=ohe_columns), df_train_encoded], axis=1)
    X_test = pd.concat([X_test.drop(columns=ohe_columns), df_test_encoded], axis=1)

    return X_train, X_test
