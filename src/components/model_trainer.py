# src/components/model_trainer.py
import os, sys, glob, joblib, numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.exceptions import CustomException
from src.logger import logging

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelTrainerConfig:
    model_out: str = os.path.join("artifacts", "model.pkl")
    preprocessor_out: str = os.path.join("artifacts", "preprocessor.pkl")


# -------------------------
# Torch MLP
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): 
        return self.net(x)


def train_mlp(Xtr, ytr, Xte, yte, epochs=50, batch_size=64, lr=1e-3):
    logging.info("Training MLP (PyTorch)")
    classes = np.unique(ytr)
    cls2id = {c: i for i, c in enumerate(classes)}
    ytr_id = np.array([cls2id[c] for c in ytr])
    yte_id = np.array([cls2id[c] for c in yte])

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr_id).long()),
        batch_size=batch_size, shuffle=True
    )
    te_tensor = torch.tensor(Xte).float()

    model = SimpleMLP(Xtr.shape[1], len(classes))
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if (ep+1) % 5 == 0:
            logging.info(f"MLP epoch {ep+1}/{epochs}, loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(te_tensor).argmax(1).numpy()
    score = f1_score(yte_id, preds, average="macro")
    logging.info(f"MLP validation f1_macro={score:.3f}")
    return model, cls2id, score


# -------------------------
# Trainer Class
# -------------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def _load_features(self, ft_dir: str):
        logging.info(f"Loading features from {ft_dir}")
        X, y = [], []
        for p in glob.glob(os.path.join(ft_dir, "*.npz")):
            d = np.load(p, allow_pickle=True)
            feat = np.concatenate([d["x_audio"], d["x_video"], d["x_text"]]).astype(np.float32)
            label = d["y"].item() if d["y"].shape == () else str(d["y"])
            X.append(feat); y.append(label)
        logging.info(f"Loaded {len(X)} feature files")
        return np.stack(X), np.array(y)

    def initiate_model_training(self, features_dir: str):
        try:
            X, y = self._load_features(features_dir)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            logging.info(f"Train={len(Xtr)} | Test={len(Xte)}")

            candidates = {}
            preprocessor_to_save = None

            # === Logistic Regression ===
            logging.info("Training Logistic Regression...")
            logreg_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
            ])
            logreg_pipe.fit(Xtr, ytr)
            preds = logreg_pipe.predict(Xte)
            score = f1_score(yte, preds, average="macro")
            logging.info(f"LogReg validation f1_macro={score:.3f}")
            candidates["LogReg"] = (logreg_pipe, score)

            # === Random Forest ===
            logging.info("Training RandomForest...")
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(Xtr, ytr)
            preds = rf.predict(Xte)
            score = f1_score(yte, preds, average="macro")
            logging.info(f"RandomForest validation f1_macro={score:.3f}")
            candidates["RandomForest"] = (rf, score)

            # === MLP ===
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
            mlp, cls2id, score = train_mlp(Xtr_s, ytr, Xte_s, yte)
            candidates["MLP"] = (
                {"model": mlp.state_dict(), "cls2id": cls2id, "input_dim": Xtr.shape[1]},
                score,
            )

            # === Late Fusion ===
            logging.info("Training Late Fusion...")
            A, V, T, y_all = [], [], [], []
            for p in glob.glob(os.path.join(features_dir, "*.npz")):
                d = np.load(p, allow_pickle=True)
                A.append(d["x_audio"].astype(np.float32))
                V.append(d["x_video"].astype(np.float32))
                T.append(d["x_text"].astype(np.float32))
                y_all.append(d["y"].item() if d["y"].shape==() else str(d["y"]))
            A, V, T, y_all = np.stack(A), np.stack(V), np.stack(T), np.array(y_all)

            Xa_tr, Xa_te, y_tr, y_te = train_test_split(A, y_all, test_size=0.2, stratify=y_all, random_state=42)
            Xv_tr, Xv_te, _, _      = train_test_split(V, y_all, test_size=0.2, stratify=y_all, random_state=42)

            clf_a = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xa_tr, y_tr)
            clf_v = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xv_tr, y_tr)

            proba_a = clf_a.predict_proba(Xa_te)
            proba_v = clf_v.predict_proba(Xv_te)
            proba_fused = (proba_a + proba_v) / 2
            preds = proba_fused.argmax(axis=1)

            # Align labels
            unique_labels = np.unique(y_tr)
            preds_labels = unique_labels[preds]

            score = f1_score(y_te, preds_labels, average="macro")
            logging.info(f"Late Fusion validation f1_macro={score:.3f}")
            candidates["LateFusion"] = ({"audio": clf_a, "video": clf_v}, score)

            # === Pick Best ===
            best_name, (best_model, best_score) = max(candidates.items(), key=lambda kv: kv[1][1])
            logging.info(f"Best model: {best_name} (f1_macro={best_score:.3f})")

            # Save best model + preprocessor
            os.makedirs(os.path.dirname(self.config.model_out), exist_ok=True)
            joblib.dump(best_model, self.config.model_out)

            if best_name == "MLP":
                joblib.dump(scaler, self.config.preprocessor_out)
            elif best_name in ["LogReg"]:
                joblib.dump(None, self.config.preprocessor_out)
            elif best_name == "RandomForest":
                joblib.dump(None, self.config.preprocessor_out)
            elif best_name == "LateFusion":
                joblib.dump(None, self.config.preprocessor_out)

            logging.info(f"Saved best model → {self.config.model_out}")
            logging.info(f"Saved preprocessor → {self.config.preprocessor_out}")

            return {
                "best_model": best_name,
                "best_score": best_score,
                "all_scores": {k: v[1] for k, v in candidates.items()}
            }

        except Exception as e:
            raise CustomException(e, sys)
