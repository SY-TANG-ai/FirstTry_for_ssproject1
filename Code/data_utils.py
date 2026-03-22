import json
import os

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
    csv_path: str = r"d:\College-Lecture\cyr_Project1\FirstTry\AdultsData\adult.csv",
    n_rows: int = 20,
):
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=column_names,
        nrows=n_rows,
        skipinitialspace=True,
    )

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    df = df.replace("?", pd.NA).dropna()

    df["income"] = df["income"].map(
        {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}
    )

    target = "income"
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numerical_features = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "fnlwgt",
    ]
    categorical_cols = [c for c in categorical_features if c in df.columns]
    numeric_cols = [c for c in numerical_features if c in df.columns]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, categorical_cols))
    else:
        ohe = None

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    features = df.drop(columns=[target])
    feature_array = preprocessor.fit_transform(features)
    feature_names = preprocessor.get_feature_names_out()
    result = pd.DataFrame(feature_array, columns=feature_names, index=df.index)

    result[target] = df[target].astype(int).values

    onehot_path = os.path.join(os.path.dirname(__file__), "OneHot.json")
    onehot_info = {}
    fitted_ohe = None
    if "cat" in preprocessor.named_transformers_:
        fitted_ohe = preprocessor.named_transformers_["cat"]

    if fitted_ohe is not None:
        for feature, categories in zip(categorical_cols, fitted_ohe.categories_):
            vectors = {}
            for idx, value in enumerate(categories):
                vector = [0] * len(categories)
                vector[idx] = 1
                vectors[str(value)] = vector
            onehot_info[feature] = {
                "categories": [str(v) for v in categories],
                "vectors": vectors,
            }

    with open(onehot_path, "w", encoding="utf-8") as f:
        json.dump(onehot_info, f, ensure_ascii=False, indent=2)

    return result


def prepare_and_save(
    csv_path: str = r"d:\College-Lecture\cyr_Project1\FirstTry\AdultsData\adult.csv",
    n_rows: int = 20,
    random_state_seed: int = 42,
    checkpoints_dir: str | None = None,
):
    processed = preprocess_data(csv_path=csv_path, n_rows=n_rows)
    X = processed.drop(columns=["income"]).to_numpy(dtype="float32")
    y = processed["income"].to_numpy(dtype="int64")

    seed = random_state_seed
    X_target, X_shadow, y_target, y_shadow = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=0.5, random_state=seed, stratify=y_target
    )
    X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
        X_shadow, y_shadow, test_size=0.5, random_state=seed, stratify=y_shadow
    )

    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    tensors = {
        "X_train_target.pt": torch.tensor(X_target_train),
        "y_train_target.pt": torch.tensor(y_target_train),
        "X_test_target.pt": torch.tensor(X_target_test),
        "y_test_target.pt": torch.tensor(y_target_test),
        "X_train_shadow.pt": torch.tensor(X_shadow_train),
        "y_train_shadow.pt": torch.tensor(y_shadow_train),
        "X_test_shadow.pt": torch.tensor(X_shadow_test),
        "y_test_shadow.pt": torch.tensor(y_shadow_test),
    }

    saved_paths = {}
    for name, tensor in tensors.items():
        path = os.path.join(checkpoints_dir, name)
        torch.save(tensor, path)
        saved_paths[name] = path

    return {
        "checkpoints_dir": checkpoints_dir,
        "paths": saved_paths,
    }
