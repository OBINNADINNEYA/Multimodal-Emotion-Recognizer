import os, os.path as op, argparse, glob, numpy as np, joblib
from typing import Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_split_features(ft_dir: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns:
      A_list, V_list, T_list stacked later per fold;
      y labels;
      dims [da, dv, dt]
    """
    A, V, T, y = [], [], [], []
    first_dims = None
    for p in glob.glob(op.join(ft_dir, "*.npz")):
        d = np.load(p, allow_pickle=True)
        xa, xv, xt = d["x_audio"], d["x_video"], d["x_text"]
        if first_dims is None:
            first_dims = [xa.shape[0], xv.shape[0], xt.shape[0]]
        # coerce label to str
        lbl = d["y"].item() if getattr(d["y"], "shape", ()) == () else str(d["y"])
        A.append(xa.astype(np.float32))
        V.append(xv.astype(np.float32))
        T.append(xt.astype(np.float32))
        y.append(str(lbl))
    return np.array(A, dtype=object), np.array(V, dtype=object), np.array(T, dtype=object), np.array(y), first_dims

def fit_modality_scalers(A_tr, V_tr, T_tr):
    sa, sv, st = StandardScaler(), StandardScaler(), StandardScaler()
    # stack to 2D for scaler
    sa.fit(np.stack(A_tr))
    sv.fit(np.stack(V_tr))
    st.fit(np.stack(T_tr))
    return sa, sv, st

def transform_and_fuse(A, V, T, sa, sv, st, w_a, w_v, w_t, dim_balance: bool, dims):
    # standardize per modality
    A_ = sa.transform(np.stack(A))
    V_ = sv.transform(np.stack(V))
    T_ = st.transform(np.stack(T))

    # optional dimensionality balancing
    if dim_balance:
        da, dv, dt = dims
        dsum = da + dv + dt
        ga = np.sqrt(dsum / max(1, da))
        gv = np.sqrt(dsum / max(1, dv))
        gt = np.sqrt(dsum / max(1, dt))
    else:
        ga = gv = gt = 1.0

    # modality weights
    A_ *= (w_a * ga)
    V_ *= (w_v * gv)
    T_ *= (w_t * gt)

    # fuse
    X = np.concatenate([A_, V_, T_], axis=1)
    return X

def build_lr(balanced: bool, max_iter: int):
    return LogisticRegression(
        max_iter=max_iter,
        n_jobs=4,
        solver="saga",
        penalty="l2",
        multi_class="multinomial",
        class_weight="balanced" if balanced else None,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--model_out", required=True)

    # evaluation modes
    ap.add_argument("--cv", type=int, default=0, help="If >1, run Stratified K-Fold CV with this many folds.")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction if --cv==0 and not training full.")
    ap.add_argument("--train_full", action="store_true", help="Train on ALL data (no holdout). Saves model.")

    # modality feature balancing
    ap.add_argument("--balance_by_dim", action="store_true",
                    help="Scale each modality by sqrt(total_dim / modality_dim).")
    ap.add_argument("--w_audio", type=float, default=1.0)
    ap.add_argument("--w_video", type=float, default=1.0)
    ap.add_argument("--w_text",  type=float, default=1.0)

    # LR args
    ap.add_argument("--balanced", action="store_true", help="Use class_weight=balanced.")
    ap.add_argument("--max_iter", type=int, default=3000)
    args = ap.parse_args()

    A, V, T, y, dims = load_split_features(args.features_dir)
    print(f"Loaded: {len(y)} samples | dims (a,v,t)={tuple(dims)} | classes={len(np.unique(y))}")

    # ===== Mode 1: K-Fold CV =====
    if args.cv and args.cv > 1:
        print(f"\n== Stratified {args.cv}-Fold CV with feature balancing ==")
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        accs, f1s = [], []

        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            sa, sv, st = fit_modality_scalers(A[tr], V[tr], T[tr])
            Xtr = transform_and_fuse(A[tr], V[tr], T[tr], sa, sv, st,
                                     args.w_audio, args.w_video, args.w_text,
                                     args.balance_by_dim, dims)
            Xte = transform_and_fuse(A[te], V[te], T[te], sa, sv, st,
                                     args.w_audio, args.w_video, args.w_text,
                                     args.balance_by_dim, dims)

            clf = build_lr(args.balanced, args.max_iter)
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            acc = accuracy_score(y[te], pred)
            f1m = f1_score(y[te], pred, average="macro")
            accs.append(acc); f1s.append(f1m)
            print(f"[Fold {fold}]  acc={acc:.4f}  macroF1={f1m:.4f}")

        print("\nCV summary:")
        print(f"  acc  mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
        print(f"  f1M  mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")

        # fit on all data with scalers from all data
        sa, sv, st = fit_modality_scalers(A, V, T)
        Xall = transform_and_fuse(A, V, T, sa, sv, st,
                                  args.w_audio, args.w_video, args.w_text,
                                  args.balance_by_dim, dims)
        clf_full = build_lr(args.balanced, args.max_iter).fit(Xall, y)

        os.makedirs(op.dirname(args.model_out), exist_ok=True)
        joblib.dump({"model": clf_full, "scalers": (sa, sv, st),
                     "weights": (args.w_audio, args.w_video, args.w_text),
                     "dims": dims, "balance_by_dim": args.balance_by_dim}, args.model_out)
        print(" saved (full-data fit) →", args.model_out)
        return

    # ===== Mode 2: Train full data =====
    if args.train_full:
        sa, sv, st = fit_modality_scalers(A, V, T)
        Xall = transform_and_fuse(A, V, T, sa, sv, st,
                                  args.w_audio, args.w_video, args.w_text,
                                  args.balance_by_dim, dims)
        clf = build_lr(args.balanced, args.max_iter).fit(Xall, y)

        os.makedirs(op.dirname(args.model_out), exist_ok=True)
        joblib.dump({"model": clf, "scalers": (sa, sv, st),
                     "weights": (args.w_audio, args.w_video, args.w_text),
                     "dims": dims, "balance_by_dim": args.balance_by_dim}, args.model_out)
        print(" saved (full-data fit) →", args.model_out)
        return

    # ===== Mode 3: Holdout =====
    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=args.test_size, random_state=42, stratify=y
    )
    sa, sv, st = fit_modality_scalers(A[tr_idx], V[tr_idx], T[tr_idx])
    Xtr = transform_and_fuse(A[tr_idx], V[tr_idx], T[tr_idx], sa, sv, st,
                             args.w_audio, args.w_video, args.w_text,
                             args.balance_by_dim, dims)
    Xte = transform_and_fuse(A[te_idx], V[te_idx], T[te_idx], sa, sv, st,
                             args.w_audio, args.w_video, args.w_text,
                             args.balance_by_dim, dims)
    clf = build_lr(args.balanced, args.max_iter).fit(Xtr, y[tr_idx])
    pred = clf.predict(Xte)
    print(classification_report(y[te_idx], pred))

    os.makedirs(op.dirname(args.model_out), exist_ok=True)
    joblib.dump({"model": clf, "scalers": (sa, sv, st),
                 "weights": (args.w_audio, args.w_video, args.w_text),
                 "dims": dims, "balance_by_dim": args.balance_by_dim}, args.model_out)
    print(" saved →", args.model_out)

if __name__ == "__main__":
    main()
