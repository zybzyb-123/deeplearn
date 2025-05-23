import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def prepare_lstm_data(trace, window_size=5):
    X, y = [], []
    for i in range(window_size, len(trace)):
        seq = [trace[i - j][1] for j in range(window_size, 0, -1)]
        target = trace[i][1]
        X.append(seq)
        y.append(target)
    return X, y

def encode_lstm_sequences(X, y):
    encoder = LabelEncoder()
    all_ids = [pid for seq in X for pid in seq] + y
    encoder.fit(all_ids)
    X_encoded = [[encoder.transform([pid])[0] for pid in seq] for seq in X]
    y_encoded = encoder.transform(y)
    vocab_size = len(encoder.classes_)
    return np.array(X_encoded), to_categorical(y_encoded, num_classes=vocab_size), encoder, vocab_size

def build_lstm_model(input_length, vocab_size, embed_dim=64, lstm_units=128):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm_model(train_trace, window_size=5):
    X_raw, y_raw = prepare_lstm_data(train_trace, window_size)
    X, y, encoder, vocab_size = encode_lstm_sequences(X_raw, y_raw)
    model = build_lstm_model(input_length=window_size, vocab_size=vocab_size)
    model.fit(X, y, epochs=3, batch_size=64, verbose=1)
    return model, encoder

def generate_lstm_topk(model, recent_trace, encoder, top_k=250):
    window_size = model.input_shape[1]
    recent_seq = [pid for _, pid in recent_trace[-window_size:]]
    encoded_seq = encoder.transform(recent_seq)
    input_seq = pad_sequences([encoded_seq], maxlen=window_size)
    probs = model.predict(input_seq, verbose=0)[0]
    top_indices = probs.argsort()[-top_k:][::-1]
    return encoder.inverse_transform(top_indices)
