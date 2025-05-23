import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ---------------------
# Trace 加载
# ---------------------
def load_trace(path):
    df = pd.read_csv(path)
    df = df[['timestamp', 'page_id']].dropna()
    trace = df[['timestamp', 'page_id']].values.tolist()
    return trace

# ---------------------
# Trace 加载与去重
# ---------------------
def load_dup_trace(path):
    df = pd.read_csv(path)
    df = df[['timestamp', 'page_id']].dropna()
    trace = df[['timestamp', 'page_id']].values.tolist()
    return remove_consecutive_duplicates(trace)

def remove_consecutive_duplicates(trace):
    result = [trace[0]]
    for i in range(1, len(trace)):
        if trace[i][1] != trace[i - 1][1]:
            result.append(trace[i])
    return result

# ---------------------
# LSTM 模型训练数据构造
# ---------------------
def prepare_lstm_data(trace, window_size=5):
    X, y = [], []
    for i in range(window_size, len(trace)):
        seq = [trace[i - j][1] for j in range(window_size, 0, -1)]
        target = trace[i][1]
        X.append(seq)
        y.append(target)
    return X, y

# ---------------------
# 编码页面 ID
# ---------------------
def encode_sequences(X, y):
    encoder = LabelEncoder()
    all_pids = [pid for seq in X for pid in seq] + y
    encoder.fit(all_pids)
    X_encoded = [[encoder.transform([pid])[0] for pid in seq] for seq in X]
    y_encoded = encoder.transform(y)
    vocab_size = len(encoder.classes_)
    return np.array(X_encoded), to_categorical(y_encoded, num_classes=vocab_size), encoder, vocab_size

# ---------------------
# 构建 LSTM 模型
# ---------------------
def build_lstm_model(input_length, vocab_size, embed_dim=64, lstm_units=128):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ---------------------
# 使用 LSTM 模型生成预取页集合
# ---------------------
def generate_lstm_prefetch(model, recent_trace, encoder, top_k=1000):
    window_size = model.input_shape[1]
    recent_seq = [pid for _, pid in recent_trace[-window_size:]]
    encoded_seq = encoder.transform(recent_seq)
    input_seq = pad_sequences([encoded_seq], maxlen=window_size)
    probs = model.predict(input_seq, verbose=0)[0]
    top_indices = probs.argsort()[-top_k:][::-1]
    return encoder.inverse_transform(top_indices)

# ---------------------
# 简单模拟缓存替换策略（LRU）
# ---------------------
class SimulatedCache:
    def __init__(self, size):
        self.size = size
        self.cache = set()
        self.queue = deque()
        self.hits = 0
        self.total = 0
        self.prefetch_hits = 0
        self.prefetched_pages = set()

    def preload(self, pages):
        for p in pages:
            self._insert(p)
            self.prefetched_pages.add(p)

    def access(self, page_id):
        self.total += 1
        if page_id in self.cache:
            self.hits += 1
            if page_id in self.prefetched_pages:
                self.prefetch_hits += 1
        else:
            self._insert(page_id)

    def _insert(self, page_id):
        if page_id in self.cache:
            return
        if len(self.cache) >= self.size:
            old = self.queue.popleft()
            self.cache.remove(old)
        self.cache.add(page_id)
        self.queue.append(page_id)

    def stats(self):
        return {
            'hit_rate': self.hits / self.total,
            'prefetch_hit_rate': self.prefetch_hits / self.total,
            'total_requests': self.total,
            'cache_hits': self.hits,
            'prefetch_hits': self.prefetch_hits
        }

# ---------------------
# 实验主函数：训练 + 预取 + 模拟
# ---------------------
def run_lstm_prefetch_experiment(train_path, test_path, cache_size=1000, top_k=1000, window_size=5):
    print("📥 加载并预处理 trace 数据...")
    train_trace = load_dup_trace(train_path)
    test_trace = load_trace(test_path)

    print("📊 构建训练样本...")
    X_raw, y_raw = prepare_lstm_data(train_trace, window_size=window_size)
    X, y, encoder, vocab_size = encode_sequences(X_raw, y_raw)

    print(f"🧠 训练 LSTM 模型（样本数: {len(X)}，页面数: {vocab_size}）")
    model = build_lstm_model(window_size, vocab_size)
    model.fit(X, y, epochs=100, batch_size=64, verbose=1)


    print("🔮 使用 LSTM 生成预取页集合...")
    prefetch_pages = generate_lstm_prefetch(model, train_trace, encoder, top_k=top_k)
    print(f"📦 生成预加载页数：{len(prefetch_pages)}")

    print("🚀 模拟冷启动缓存加载过程...")
    cache = SimulatedCache(cache_size)
    cache.preload(prefetch_pages)

    hit_curve = []
    for i, (_, page_id) in enumerate(test_trace):
        cache.access(page_id)
        if (i + 1) % 100 == 0:
            stats = cache.stats()
            hit_curve.append(stats['hit_rate'])

    final_stats = cache.stats()
    print("✅ 模拟完成，结果如下：", final_stats)

    plt.figure(figsize=(10, 5))
    plt.plot(hit_curve, label='LSTM 预加载命中率')
    plt.title("LSTM 缓存预取 - 命中率变化曲线")
    plt.xlabel("请求数（每 100）")
    plt.ylabel("命中率")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return final_stats

run_lstm_prefetch_experiment("train.txt", "warm.txt", 1000, 250,5)
