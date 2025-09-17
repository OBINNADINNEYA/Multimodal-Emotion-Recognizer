import os, os.path as op, argparse, glob, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_features(ft_dir: str):
    X, y = [], []
    for p in glob.glob(op.join(ft_dir,"*.npz")):
        d = np.load(p, allow_pickle=True)
        xa, xv, xt = d["x_audio"], d["x_video"], d["x_text"]
        x = np.concatenate([xa, xv, xt]).astype(np.float32)
        X.append(x)
        y.append(d["y"].item() if d["y"].shape==() else str(d["y"]))
    return np.stack(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--model_out", required=True)
    args = ap.parse_args()

    X, y = load_features(args.features_dir)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, n_jobs=4))
    ])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(classification_report(yte, pred))

    os.makedirs(op.dirname(args.model_out), exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(" saved →", args.model_out)

if __name__ == "__main__":
    main()
