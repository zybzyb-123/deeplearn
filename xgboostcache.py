# Re-run the code after kernel reset to restore imports and function definitions

import pandas as pd
import numpy as np
from collections import deque, defaultdict
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------
# Step 1: Load and deduplicate trace
# ---------------------
def load_trace(path):
    df = pd.read_csv(path)
    df = df[['timestamp', 'page_id']].dropna()
    trace = df[['timestamp', 'page_id']].values.tolist()
    return remove_consecutive_duplicates(trace)

def remove_consecutive_duplicates(trace):
    if not trace:
        return []
    result = [trace[0]]
    for i in range(1, len(trace)):
        if trace[i][1] != trace[i-1][1]:
            result.append(trace[i])
    return result

# ---------------------
# Step 2: Prepare range-based training data
# ---------------------
def generate_range_training_data(trace, window_size=3, future_window=10):
    X, y_a, y_b = [], [], []
    for i in range(window_size, len(trace) - future_window):
        prev_pages = [trace[i - j][1] for j in range(window_size, 0, -1)]
        future_pages = [trace[i + j][1] for j in range(future_window)]
        a = min(future_pages)
        b = max(future_pages)
        X.append(prev_pages)
        y_a.append(a)
        y_b.append(b)
    return np.array(X), np.array(y_a), np.array(y_b)

# ---------------------
# Step 3: Train regressors for a and b
# ---------------------
def train_range_models(X, y_a, y_b):
    model_a = XGBRegressor()
    model_b = XGBRegressor()
    model_a.fit(X, y_a)
    model_b.fit(X, y_b)
    return model_a, model_b

# ---------------------
# Step 4: Generate dynamic prefetch page set using [a, b]
# ---------------------
def generate_xgboost_range_prefetch(model_a, model_b, trace, alpha_limit=12, top_k=1000):
    prefetch_pages = set()
    for i in range(3, len(trace)):
        window = [trace[i - j][1] for j in range(3, 0, -1)]
        X = np.array(window).reshape(1, -1)

        a = int(model_a.predict(X)[0])
        b = int(model_b.predict(X)[0])
        if a > b:
            a, b = b, a

        if b - a + 1 > alpha_limit:
            b = a + alpha_limit - 1

        for pid in range(a, b + 1):
            prefetch_pages.add(pid)

        if len(prefetch_pages) >= top_k:
            break
    print (list(prefetch_pages)[:top_k])
    return list(prefetch_pages)[:top_k]

# ---------------------
# Step 5: Simulated Cache (LRU)
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
# Step 6: Run experiment with prefetch + test trace
# ---------------------
def run_range_prefetch_experiment(train_trace, test_trace, cache_size=1000, alpha_limit=12, top_k=250):
    print("📥 加载和重复数据删除跟踪...")
    train_trace = remove_consecutive_duplicates(train_trace)

    print("📊 准备训练数据...")
    X, y_a, y_b = generate_range_training_data(train_trace)
    model_a, model_b = train_range_models(X, y_a, y_b)

    print("🔮 生成基于范围的预取页面...")
    prefetch_pages = generate_xgboost_range_prefetch(model_a, model_b, train_trace, alpha_limit, top_k)
    print(f"📦 生成的预取页面: {len(prefetch_pages)}")

    print("🚀 模拟带预取的缓存...")
    cache = SimulatedCache(cache_size)
    cache.preload(prefetch_pages)

    hit_curve = []
    for i, (_, page_id) in enumerate(test_trace):
        cache.access(page_id)
        if (i + 1) % 100 == 0:
            stats = cache.stats()
            hit_curve.append(stats['hit_rate'])

    final_stats = cache.stats()
    print("✅ Final stats:", final_stats)

    plt.figure(figsize=(10, 5))
    plt.plot(hit_curve, label='Hit Rate')
    plt.title("XGBoost Range-Based Prefetching - Cache Hit Rate")
    plt.xlabel("Requests (x100)")
    plt.ylabel("Hit Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return final_stats

if __name__ == '__main__':
    print("📥 加载 trace 数据...")
    trace_file = 'moodle_trace7_8kb.txt'
    trace = []
    with open(trace_file, 'r') as f:
        for line in f:
            timestamp, page_id = line.strip().split(',')
            timestamp = float(timestamp)
            page_id = int(page_id)
            trace.append((timestamp, page_id))

    #12-13=38477
    train_trace = trace[:38477]
    warm_trace = trace[:38477]
    test_trace = trace[38477:53476]

    run_range_prefetch_experiment(warm_trace,test_trace,1000,10,250)