# src/infer.py
import argparse
import torch
import pandas as pd
from src.data.preprocess import load_and_preprocess
from src.models.models import LSTMForecaster
import os

def infer(input_path, output_path, checkpoint="models/checkpoint.pth"):
    X, y, scaler = load_and_preprocess(input_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_dim=X.shape[2])
    if not os.path.exists(checkpoint):
        print("Model checkpoint not found. Run train.py first or provide a checkpoint.")
        return
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    import numpy as np
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(Xt).cpu().numpy()
    df_out = pd.DataFrame({"pred_normalized": preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/sample_load.csv")
    parser.add_argument("--output", default="out/predictions.csv")
    parser.add_argument("--model", default="models/checkpoint.pth")
    args = parser.parse_args()
    infer(args.input, args.output, args.model)
