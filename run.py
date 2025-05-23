import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from lstm_prefetch import train_lstm_model, generate_lstm_topk
from xgboost_prefetch import train_xgboost_model, generate_xgb_topk
from lru_baseline import get_lru_topk_by_simulation

def load_trace_dup(path):
    df = pd.read_csv(path)
    trace = df[['timestamp', 'page_id']].dropna().values.tolist()
    return remove_consecutive_duplicates(trace)

def load_trace(path):
    df = pd.read_csv(path)
    trace = df[['timestamp', 'page_id']].dropna().values.tolist()
    return trace

def remove_consecutive_duplicates(trace):
    result = [trace[0]]
    for i in range(1, len(trace)):
        if trace[i][1] != trace[i - 1][1]:
            result.append(trace[i])
    return result

class SimulatedCache:
    def __init__(self, size):
        self.size = size
        self.cache = set()
        self.queue = deque()
        self.hits = 0
        self.total = 0

    def preload(self, pages):
        for p in pages:
            self._insert(p)

    def access(self, page_id):
        self.total += 1
        if page_id in self.cache:
            self.hits += 1
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
        return self.hits / self.total

def simulate_cache(trace, preload_pages, cache_size):
    cache = SimulatedCache(cache_size)
    cache.preload(preload_pages)
    hit_curve = []
    for i, (_, pid) in enumerate(trace):
        cache.access(pid)
        if (i + 1) % 100 == 0:
            hit_curve.append(cache.hits / cache.total)
    return hit_curve, cache.stats()

def plot_hit_rate_curves(curves):
    plt.figure(figsize=(10, 5))
    for label, curve in curves.items():
        plt.plot(curve, label=label)
    plt.title("缓存命中率对比")
    plt.xlabel("请求数（每100）")
    plt.ylabel("命中率")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_trace = load_trace("train.txt")
    test_trace = load_trace("warm.txt")
    cache_size = 1000
    top_k = 250

    print("训练 LSTM...")
    lstm_model, lstm_encoder = train_lstm_model(train_trace)
    lstm_prefetch = generate_lstm_topk(lstm_model, train_trace, lstm_encoder, top_k)
    lstm_curve, lstm_hit = simulate_cache(test_trace, lstm_prefetch, cache_size)

    print("训练 XGBoost...")
    xgb_model, xgb_encoder = train_xgboost_model(train_trace)
    xgb_prefetch = generate_xgb_topk(xgb_model, train_trace, xgb_encoder, top_k)
    xgb_curve, xgb_hit = simulate_cache(test_trace, xgb_prefetch, cache_size)

    print("使用 LRU 快照...")
    lru_prefetch = get_lru_topk_by_simulation(train_trace, top_k)
    lru_curve, lru_hit = simulate_cache(test_trace, lru_prefetch, cache_size)

    print("命中率比较：")
    print(f"LSTM: {lstm_hit:.4f}")
    print(f"XGBoost: {xgb_hit:.4f}")
    print(f"LRU 快照: {lru_hit:.4f}")

    plot_hit_rate_curves({
        "LSTM": lstm_curve,
        "XGBoost": xgb_curve,
        "LRU 快照": lru_curve
    })
