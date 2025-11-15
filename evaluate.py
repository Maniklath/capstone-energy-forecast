# src/evaluate.py
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(gt_path, pred_path):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)

    # align lengths
    y_true = gt['load_kw'].values[-len(pred):]
    y_pred = pred.iloc[:,0].values

    mae = mean_absolute_error(y_true, y_pred)

    # FIX: compute RMSE manually so sklearn version does not matter
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mape = (abs((y_true - y_pred) / (y_true + 1e-9))).mean() * 100

    print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}

if __name__=="__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate.py <ground_truth_csv> <predictions_csv>")
    else:
        evaluate(sys.argv[1], sys.argv[2])
