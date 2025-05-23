import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def prepare_xgb_data(trace, window_size=3):
    X, y = [], []
    for i in range(window_size, len(trace)):
        seq = [trace[i - j][1] for j in range(window_size, 0, -1)]
        target = trace[i][1]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def train_xgboost_model(trace, window_size=3):
    X_raw, y_raw = prepare_xgb_data(trace, window_size)
    encoder = LabelEncoder()
    all_ids = np.unique(np.concatenate([X_raw.flatten(), y_raw]))
    encoder.fit(all_ids)
    X = np.array([[encoder.transform([p])[0] for p in seq] for seq in X_raw])
    y = encoder.transform(y_raw)
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model, encoder

def generate_xgb_topk(model, recent_trace, label_encoder, top_k=250):
    window_size = len(model.feature_names_in_)
    recent_seq = [pid for _, pid in recent_trace[-window_size:]]
    encoded_seq = label_encoder.transform(recent_seq).reshape(1, -1)
    probs = model.predict_proba(encoded_seq)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return label_encoder.inverse_transform(top_indices)
