import joblib, numpy as np

def predict(sample_features, model_path="models/transformer.pkl"):
    model = joblib.load(model_path)
    pred = model.predict([sample_features])
    return pred[0]

if __name__=="__main__":
    # example: load npz for one sample
    d = np.load("data/features/test3/sample_00001.npz", allow_pickle=True)
    x = np.concatenate([d["x_audio"], d["x_video"]])
    print("Predicted:", predict(x))
