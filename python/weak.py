import os
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kmeans_parallel import KMeans

def main():
    base_train_path = "../data/movielens1m.csv"
    BASE_ROWS_PER_THREAD = 1000
    NUM_RUNS = 30
    MAX_THREADS = 12
    K_CLUSTERS = 3
    MAX_ITER = 5
    OUTPUT_DIR = "weak_scaling"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running Python Weak Scaling (K={K_CLUSTERS}, Max Threads={MAX_THREADS}, Runs={NUM_RUNS})...")

    df_full = pd.read_csv(base_train_path)
    total_rows_available = len(df_full)

    for t in range(1, MAX_THREADS + 1):
        num_rows = BASE_ROWS_PER_THREAD * t
        if num_rows > total_rows_available:
            print(f"Not enough rows for {t} threads ({num_rows} > {total_rows_available}), skipping.")
            break

        output_file_name = os.path.join(OUTPUT_DIR, f"kmeans_n{t}_results.csv")

        print(f"\nThreads: {t}, Data size: {num_rows}")

        sum_Ts_work = 0.0
        sum_Tp_work = 0.0
        sum_T_total = 0.0

        df_train = df_full.iloc[:num_rows]
        X_train = df_train.to_numpy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        try:
            with open(output_file_name, 'w') as file:
                file.write("Run,TotalTime_s,SerialTime_s,ParallelTime_s\n")

                for i in range(1, NUM_RUNS + 1):
                    model = KMeans(K_CLUSTERS, MAX_ITER, t)

                    t_fit_total, t_fit_serial, t_fit_parallel = model.fit(X_train)

                    t_fit_total_s = t_fit_total.total_seconds()
                    t_fit_serial_s = t_fit_serial.total_seconds()
                    t_fit_parallel_s = t_fit_parallel.total_seconds()

                    sum_Ts_work += t_fit_serial_s
                    sum_Tp_work += t_fit_parallel_s
                    sum_T_total += t_fit_total_s

                    file.write(f"{i}, {t_fit_total_s:.6f}, {t_fit_serial_s:.6f}, {t_fit_parallel_s:.6f}\n")

                avg_Ts_work = sum_Ts_work / NUM_RUNS
                avg_Tp_work = sum_Tp_work / NUM_RUNS
                avg_T_total = sum_T_total / NUM_RUNS
                total_ideal_work = avg_Ts_work + avg_Tp_work

                file.write(f"\n=== Statistics (Threads={t}) ===\n")
                file.write(f"Average Wall-Clock Time: {avg_T_total:.4f} s\n")
                file.write(f"Average ideal work: {total_ideal_work:.4f} s\n")

            print(f"Completed Threads={t}. Results saved to {output_file_name}")

        except Exception as e:
            print(f"Error with Threads={t}: {e}")

if __name__ == "__main__":
    main()
