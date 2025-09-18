import os, os.path as op, argparse, glob, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Torch for MLP + Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Data loading
# -------------------------
def load_features(ft_dir: str, use_text: bool):
    A, V, T, y = [], [], [], []
    for p in glob.glob(op.join(ft_dir,"*.npz")):
        d = np.load(p, allow_pickle=True)
        xa, xv = d["x_audio"], d["x_video"]
        xt = d["x_text"] if use_text and "x_text" in d else np.zeros(1, dtype=np.float32)
        A.append(xa.astype(np.float32))
        V.append(xv.astype(np.float32))
        T.append(xt.astype(np.float32))
        y.append(d["y"].item() if d["y"].shape==() else str(d["y"]))
    return np.stack(A), np.stack(V), np.stack(T), np.array(y)

# -------------------------
# Models
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class TinyTransformer(nn.Module):
    def __init__(self, d_model, num_classes, nhead=2, num_layers=1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens):  # tokens shape [batch, 3, d_model]
        x = tokens.permute(1,0,2)  # [seq,batch,dim]
        out = self.encoder(x)      # same shape
        pooled = out.mean(0)       # [batch,dim]
        return self.cls(pooled)

# -------------------------
# Training helpers
# -------------------------
def train_torch_model(model, Xtr, ytr, Xte, yte, epochs=50, batch_size=64, lr=1e-3):
    classes = np.unique(ytr)
    cls2id = {c:i for i,c in enumerate(classes)}
    ytr_id = np.array([cls2id[c] for c in ytr])
    yte_id = np.array([cls2id[c] for c in yte])

    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr_id).long()),
                           batch_size=batch_size, shuffle=True)
    te_tensor = torch.tensor(Xte).float()
    te_labels = torch.tensor(yte_id).long()

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if (ep+1)%5==0: print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(te_tensor).argmax(1).numpy()
    report = classification_report(yte_id, preds, target_names=classes)
    return model, cls2id, report

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--model_type", choices=["logreg","mlp","transformer","latefusion"], default="logreg")
    ap.add_argument("--use_text", action="store_true")
    args = ap.parse_args()

    A, V, T, y = load_features(args.features_dir, args.use_text)
    # concatenate if needed
    if args.model_type in ["logreg","mlp"]:
        X = np.concatenate([A,V,T], axis=1) if args.use_text else np.concatenate([A,V], axis=1)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ===== LogReg =====
    if args.model_type=="logreg":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, n_jobs=4, class_weight="balanced"))
        ])
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        print(classification_report(yte, pred))
        joblib.dump(clf, args.model_out)

    # ===== MLP =====
    elif args.model_type=="mlp":
        Xtr = StandardScaler().fit_transform(Xtr)
        Xte = StandardScaler().fit_transform(Xte)
        model, cls2id, report = train_torch_model(SimpleMLP(Xtr.shape[1], len(np.unique(y))), Xtr, ytr, Xte, yte)
        print(report)
        joblib.dump({"model":model.state_dict(),"cls2id":cls2id,"input_dim":Xtr.shape[1]}, args.model_out)

    # ===== Transformer Fusion =====
    elif args.model_type=="transformer":
        # make tokens = [audio, video, text] → pad dims to same size
        d_model = max(A.shape[1], V.shape[1], T.shape[1])
        def pad(x): return np.pad(x, (0,d_model-x.shape[0]))
        tokens = np.stack([np.apply_along_axis(pad,1,A),
                           np.apply_along_axis(pad,1,V),
                           np.apply_along_axis(pad,1,T)], axis=1)  # [N,3,d_model]
        Xtr, Xte, ytr, yte = train_test_split(tokens, y, test_size=0.2, random_state=42, stratify=y)
        model, cls2id, report = train_torch_model(TinyTransformer(d_model,len(np.unique(y))), Xtr, ytr, Xte, yte)
        print(report)
        joblib.dump({"model":model.state_dict(),"cls2id":cls2id,"d_model":d_model}, args.model_out)

    # ===== Late Fusion =====
    elif args.model_type=="latefusion":
        def fit_lr(Xtr,ytr): 
            pipe = Pipeline([("scaler",StandardScaler()),("lr",LogisticRegression(max_iter=2000,class_weight="balanced"))])
            pipe.fit(Xtr,ytr); return pipe
        Xa,Xv,Xt = A,V,T
        Xa_tr,Xa_te,ya_tr,ya_te = train_test_split(Xa,y,test_size=0.2,random_state=42,stratify=y)
        Xv_tr,Xv_te,yv_tr,yv_te = train_test_split(Xv,y,test_size=0.2,random_state=42,stratify=y)
        clf_a,clf_v = fit_lr(Xa_tr,ya_tr), fit_lr(Xv_tr,yv_tr)
        preds = (clf_a.predict_proba(Xa_te) + clf_v.predict_proba(Xv_te))/2
        ypred = preds.argmax(1)
        print(classification_report(yv_te, ypred))
        joblib.dump({"audio":clf_a,"video":clf_v}, args.model_out)

    print("saved →", args.model_out)

if __name__=="__main__":
    main()
