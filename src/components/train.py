"""Component 5: Model Training.

Per A4 Task 3: XGBoost / LightGBM / RF+FS hybrid.
Per A4 Task 4: Cost-sensitive learning (high FN penalty via scale_pos_weight
or class_weight depending on model).
"""
from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(base_image="mlops-a4-base:v1")
def train_model(
    train_fe: Input[Dataset],
    model_out: Output[Model],
    model_type: str = "xgboost",
    cost_sensitive: bool = False,
    fn_penalty: float = 5.0,
    n_estimators: int = 200,
    max_depth: int = 6,
    random_state: int = 42,
) -> str:
    import json
    import pickle  # standard for sklearn model serialization

    import pandas as pd

    df = pd.read_csv(train_fe.path)
    y = df["isFraud"].values.astype(int)
    X = df.drop(columns=["isFraud"]).values

    pos = max(int(y.sum()), 1)
    neg = len(y) - pos
    base_pos_weight = neg / pos
    pos_weight = base_pos_weight * fn_penalty if cost_sensitive else 1.0

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.1, random_state=random_state, n_jobs=1,
            scale_pos_weight=pos_weight, eval_metric="auc",
            tree_method="hist",
        ).fit(X, y)
        params = {"n_estimators": n_estimators, "max_depth": max_depth,
                  "learning_rate": 0.1, "scale_pos_weight": float(pos_weight)}
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, random_state=random_state, n_jobs=1,
            scale_pos_weight=pos_weight,
            verbose=-1,
        ).fit(X, y)
        params = {"n_estimators": n_estimators, "max_depth": max_depth,
                  "learning_rate": 0.05, "scale_pos_weight": float(pos_weight)}
    elif model_type == "rf_fs":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE
        base = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1)
        n_keep = min(20, X.shape[1])
        sel = RFE(base, n_features_to_select=n_keep).fit(X, y)
        X_sel = sel.transform(X)
        cw = {0: 1.0, 1: float(pos_weight)} if cost_sensitive else "balanced"
        model_final = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=1,
            class_weight=cw,
        ).fit(X_sel, y)

        class _RFEPipeline:
            def __init__(self, sel, model): self.sel, self.model = sel, model
            def predict(self, X): return self.model.predict(self.sel.transform(X))
            def predict_proba(self, X): return self.model.predict_proba(self.sel.transform(X))
        model = _RFEPipeline(sel, model_final)
        params = {"n_estimators": n_estimators, "max_depth": max_depth,
                  "n_features_selected": int(n_keep), "class_weight": str(cw)}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    bundle = {
        "model": model,
        "model_type": model_type,
        "cost_sensitive": cost_sensitive,
        "fn_penalty": float(fn_penalty),
        "params": params,
    }
    with open(model_out.path, "wb") as f:
        pickle.dump(bundle, f)

    summary = {**params, "model_type": model_type, "cost_sensitive": cost_sensitive,
               "n_train": int(len(y)), "train_fraud_rate": float(y.mean())}
    print(f"[TRAIN] {summary}")
    return json.dumps(summary)
