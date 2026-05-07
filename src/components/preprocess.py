"""Component 3: Data Preprocessing.

Handles per A4 Task 2:
  - Missing values (median for numeric, mode for categorical)
  - High-cardinality categoricals (target encoding for card1; one-hot for low-card)
  - Time-based train/test split (per A4 Task 7 drift)
  - Class imbalance strategy (none / smote / undersample / class_weight)
"""
from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(base_image="mlops-a4-base:v1")
def preprocess_data(
    validated: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    imbalance_strategy: str = "none",  # none | smote | undersample | class_weight
    random_state: int = 42,
    time_based_split: bool = False,    # if True, train on earlier TransactionDT
) -> str:
    import json

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(validated.path)
    y = df["isFraud"].values.astype(int)
    X = df.drop(columns=["isFraud"]).copy()

    # ---------- impute ----------
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "MISSING")

    # ---------- target encoding for high-cardinality ----------
    high_card = [c for c in cat_cols if X[c].nunique() > 30]
    low_card = [c for c in cat_cols if c not in high_card]
    for c in high_card:
        # smoothed target encoding: (sum + alpha*global) / (count + alpha)
        global_mean = y.mean()
        alpha = 10.0
        agg = pd.DataFrame({"y": y, "x": X[c].values}).groupby("x")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)
        X[c + "_te"] = X[c].map(enc).fillna(global_mean)
        X = X.drop(columns=[c])

    # ---------- one-hot for low-card ----------
    if low_card:
        X = pd.get_dummies(X, columns=low_card, dummy_na=False, drop_first=True)

    # ---------- split ----------
    if time_based_split and "TransactionDT" in X.columns:
        # earlier 1 - test_size for train
        order = X["TransactionDT"].argsort().values
        cut = int(len(X) * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    # ---------- imbalance handling ----------
    if imbalance_strategy == "smote":
        from imblearn.over_sampling import SMOTE
        # SMOTE only on training
        X_train_np = X_train.select_dtypes(include=[np.number]).values
        feat_cols = X_train.select_dtypes(include=[np.number]).columns
        sm = SMOTE(random_state=random_state, sampling_strategy=0.30)  # raise minority to 30%
        X_train_res, y_train = sm.fit_resample(X_train_np, y_train)
        X_train = pd.DataFrame(X_train_res, columns=feat_cols)
        # recompute X_test with same numeric cols
        X_test = X_test[feat_cols]
    elif imbalance_strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=random_state, sampling_strategy=0.30)
        X_train_np = X_train.select_dtypes(include=[np.number]).values
        feat_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_res, y_train = rus.fit_resample(X_train_np, y_train)
        X_train = pd.DataFrame(X_train_res, columns=feat_cols)
        X_test = X_test[feat_cols]
    elif imbalance_strategy == "class_weight":
        # No resampling — weight is applied in training step
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test[X_train.columns]
    else:
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test[X_train.columns]

    # ---------- write artifacts ----------
    pd.concat([X_train.reset_index(drop=True),
               pd.Series(y_train, name="isFraud").reset_index(drop=True)], axis=1) \
      .to_csv(train_data.path, index=False)
    pd.concat([X_test.reset_index(drop=True),
               pd.Series(y_test, name="isFraud").reset_index(drop=True)], axis=1) \
      .to_csv(test_data.path, index=False)

    summary = {
        "imbalance_strategy": imbalance_strategy,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "train_fraud_rate": float(np.mean(y_train)),
        "test_fraud_rate": float(np.mean(y_test)),
        "high_card_target_encoded": high_card,
    }
    print(f"[PREPROCESS] {summary}")
    return json.dumps(summary)
