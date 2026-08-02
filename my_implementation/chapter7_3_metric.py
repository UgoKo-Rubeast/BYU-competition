import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "tomo_id",
    "Motor axis 0",
    "Motor axis 1",
    "Motor axis 2",
]


def _validate_required_columns(df: pd.DataFrame, required_cols, df_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def distance_metric(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    thresh_ratio: float,
    min_radius: float,
):
    """Return binary predictions after distance-based positive filtering."""
    coordinate_cols = ["Motor axis 0", "Motor axis 1", "Motor axis 2"]

    labels = solution[coordinate_cols].to_numpy(dtype=np.float32)
    preds = submission[coordinate_cols].to_numpy(dtype=np.float32)
    distances = np.linalg.norm(labels - preds, axis=1)

    thresholds = (float(min_radius) * float(thresh_ratio)) / solution["Voxel spacing"].to_numpy(dtype=np.float32)
    pred_has_motor = submission["Has motor"].to_numpy(dtype=np.int64).copy()
    gt_has_motor = solution["Has motor"].to_numpy(dtype=np.int64)

    # If both GT/pred are positive but predicted center is too far, invalidate the prediction.
    far_pos = (distances > thresholds) & (gt_has_motor == 1) & (pred_has_motor == 1)
    pred_has_motor[far_pos] = 0
    return pred_has_motor


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    beta = float(beta)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta * beta
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0

    return {
        "fbeta": float(fbeta),
        "precision": float(precision),
        "recall": float(recall),
    }


def score(solution: pd.DataFrame, submission: pd.DataFrame, min_radius: float, beta: float = 2.0):
    """Compute competition-style metrics with distance-aware positive matching."""
    _validate_required_columns(solution, REQUIRED_COLUMNS + ["Voxel spacing", "Has motor"], "solution")
    _validate_required_columns(submission, REQUIRED_COLUMNS, "submission")

    solution = solution.sort_values("tomo_id").reset_index(drop=True).copy()
    submission = submission.sort_values("tomo_id").reset_index(drop=True).copy()

    if len(solution) != len(submission):
        raise ValueError("solution/submission lengths do not match")
    if not solution["tomo_id"].equals(submission["tomo_id"]):
        raise ValueError("Submitted tomo_id values do not match the solution file")

    submission["Has motor"] = 1
    no_motor_mask = (submission[["Motor axis 0", "Motor axis 1", "Motor axis 2"]] == -1).any(axis=1)
    submission.loc[no_motor_mask, "Has motor"] = 0

    predictions = distance_metric(
        solution=solution,
        submission=submission,
        thresh_ratio=1.0,
        min_radius=min_radius,
    )
    y_true = solution["Has motor"].to_numpy(dtype=np.int64)
    return binary_classification_metrics(y_true, predictions, beta=beta)


def run_section_7_3_assertions():
    solution = pd.DataFrame(
        {
            "tomo_id": [0, 1, 2, 3],
            "Motor axis 0": [-1, 250, 100, 200],
            "Motor axis 1": [-1, 250, 100, 200],
            "Motor axis 2": [-1, 250, 100, 200],
            "Voxel spacing": [10, 10, 10, 10],
            "Has motor": [0, 1, 1, 1],
        }
    )
    submission = pd.DataFrame(
        {
            "tomo_id": [0, 1, 2, 3],
            "Motor axis 0": [100, 251, 600, -1],
            "Motor axis 1": [100, 251, 600, -1],
            "Motor axis 2": [100, 251, 600, -1],
        }
    )

    out = score(solution, submission, min_radius=1000, beta=2.0)
    assert abs(out["precision"] - 0.5) < 1e-8
    assert abs(out["recall"] - (1.0 / 3.0)) < 1e-8
    assert abs(out["fbeta"] - (5.0 / 14.0)) < 1e-8

    # Distance filter check: close positive stays positive, far positive is invalidated.
    s2 = pd.DataFrame(
        {
            "tomo_id": [10, 11],
            "Motor axis 0": [0, 0],
            "Motor axis 1": [0, 0],
            "Motor axis 2": [0, 0],
            "Voxel spacing": [10, 10],
            "Has motor": [1, 1],
        }
    )
    p2 = pd.DataFrame(
        {
            "tomo_id": [10, 11],
            "Motor axis 0": [1, 500],
            "Motor axis 1": [1, 500],
            "Motor axis 2": [1, 500],
        }
    )
    p2["Has motor"] = [1, 1]
    preds2 = distance_metric(s2, p2, thresh_ratio=1.0, min_radius=100)
    assert int(preds2[0]) == 1
    assert int(preds2[1]) == 0

    print("[5-3/metric_example]", out)
    print("[5-3/distance_filter_preds]", preds2.tolist())
