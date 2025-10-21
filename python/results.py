from sklearn.discriminant_analysis import StandardScaler
from kmeans_parallel import KMeans
import pandas as pd

if __name__ == "__main__":
    base_train_path = "../data/movielens1m.csv"
    df_full = pd.read_csv(base_train_path,  nrows=10000)
    X_train = df_full.to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=5, max_iter=1000, n_threads=12)
    kmeans.fit(X_train)
