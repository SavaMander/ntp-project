import numpy as np
import random
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Cluster(object):
    def __init__(self, center):
        self.center = center
        self.data = []

    def recalculate_center(self):
        if not self.data:
            return 
        data_np = np.array(self.data)
        sum_of_data = np.sum(data_np, axis=0)
        n = len(self.data)
        self.center = (sum_of_data / n).tolist()

class KMeans(object):
    def __init__(self, n_clusters, max_iter):
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data):
        self.data = data
        dimensions = len(self.data[0])

        serial_time = 0.0
        parallel_time = 0.0  # will be 0 here, just for completeness

        # --- Initialize clusters (serial) ---
        t0 = time.time()
        for i in range(self.n_clusters):
            point = [random.random() for _ in range(dimensions)]
            self.clusters.append(Cluster(point))
        t1 = time.time()
        serial_time += (t1 - t0)

        iter_no = 0
        tolerance = 1e-4

        while iter_no < self.max_iter:
            # --- Clear clusters (serial) ---
            t0 = time.time()
            for cluster in self.clusters:
                cluster.data = []
            t1 = time.time()
            serial_time += (t1 - t0)

            # --- E-step: Assign points to clusters (serial) ---
            t0 = time.time()
            for point in self.data:
                cluster_index = self.predict(point)
                self.clusters[cluster_index].data.append(point)
            t1 = time.time()
            parallel_time += (t1 - t0)

            # --- M-step: Recalculate centers (serial) ---
            old_centers = [np.array(c.center) for c in self.clusters]
            t0 = time.time()
            for cluster in self.clusters:
                cluster.recalculate_center()
            t1 = time.time()
            parallel_time += (t1 - t0)

            # --- Convergence check (serial) ---
            t0 = time.time()
            moved_distance_sum = 0
            for i in range(self.n_clusters):
                new_center = np.array(self.clusters[i].center)
                old_center = old_centers[i]
                distance = np.linalg.norm(new_center - old_center)
                moved_distance_sum += distance
            t1 = time.time()
            serial_time += (t1 - t0)

            if moved_distance_sum < tolerance:
                break
            iter_no += 1

        total_time = serial_time + parallel_time
        f_s = serial_time / total_time
        return f_s

    def predict(self, point):
        min_distance = None
        cluster_index = None
        for index, cluster in enumerate(self.clusters):
            distance = self.euclidean_distance(point, cluster.center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance
        return cluster_index

    def euclidean_distance(self, x, y):
        return np.linalg.norm(np.array(y) - np.array(x))

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # Generate a small random dataset
    base_train_path = "../data/movielens1m.csv"
    df_full = pd.read_csv(base_train_path,  nrows=10000)
    X_train = df_full.to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=3, max_iter=10)
    f_s = kmeans.fit(X_train)

    print(f"Estimated fraction of serial code (f_s): {f_s*100:.2f}%")
