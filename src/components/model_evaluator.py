import os, os.path as op, argparse, glob, numpy as np, joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def load_features(ft_dir: str):
    X, y = [], []
    for p in glob.glob(op.join(ft_dir,"*.npz")):
        d = np.load(p, allow_pickle=True)
        x = np.concatenate([d["x_audio"], d["x_video"], d["x_text"]]).astype(np.float32)
        y.append(d["y"].item() if d["y"].shape==() else str(d["y"]))
        X.append(x)
    return np.stack(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--report_out", default="reports/report.txt")
    args = ap.parse_args()

    os.makedirs(op.dirname(args.report_out), exist_ok=True)
    X, y = load_features(args.features_dir)
    clf = joblib.load(args.model)
    pred = clf.predict(X)

    with open(args.report_out,"w") as f:
        f.write(classification_report(y, pred))

    cm = confusion_matrix(y, pred, labels=sorted(set(y)))
    pd.DataFrame(cm, index=sorted(set(y)), columns=sorted(set(y))).to_csv("reports/confusion_matrix.csv")
    print("✅ report →", args.report_out)

if __name__ == "__main__":
    main()
