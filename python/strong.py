import os
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from kmeans_parallel import KMeans # Pretpostavljamo da je izmenjeni kod u ovom fajlu

def main():
    # --- Konfiguracija Eksperimenta ---
    train_path = "../data/movielens1m.csv"
    NUM_ROWS_LIMIT = 500000 # Ograničavanje skupa podataka (kao u Rust implementaciji)
    NUM_RUNS = 5
    MAX_THREADS = 12
    K_CLUSTERS = 3 
    MAX_ITER = 5 
    OUTPUT_DIR = "strong_scaling"
    
    # --- Priprema Podataka ---
    print("Loading and Scaling data...")
    # Učitavanje i ograničavanje redova
    df_train = pd.read_csv(train_path, nrows=NUM_ROWS_LIMIT)
    X_train = df_train.to_numpy() # Konverzija u numpy niz za KMeans
    
    # Skaliranje
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # --- Strong Scaling Eksperiment ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running Python Strong Scaling (K={K_CLUSTERS}, Max Threads={MAX_THREADS}, Runs={NUM_RUNS})...")
    
    for t in range(1, MAX_THREADS + 1):
        output_file_name = os.path.join(OUTPUT_DIR, f"kmeans_n{t}_results.csv")
        
        sum_Ts_work = 0.0
        sum_Tp_work = 0.0
        sum_T_total = 0.0

        try:
            with open(output_file_name, 'w') as file:
                file.write("Run,TotalTime_s,SerialTime_s,ParallelTime_s\n")
                
                for i in range(1, NUM_RUNS + 1):
                    # Inicijalizacija modela sa brojem niti 't'
                    model = KMeans(K_CLUSTERS, MAX_ITER, t) 
                    
                    # model.fit sada vraća (total, serial, parallel) timedelta objekte
                    t_fit_total, t_fit_serial, t_fit_parallel = model.fit(X_train)

                    t_fit_total_s = t_fit_total.total_seconds()
                    t_fit_serial_s = t_fit_serial.total_seconds()
                    t_fit_parallel_s = t_fit_parallel.total_seconds()
                    
                    # Akumulacija
                    sum_Ts_work += t_fit_serial_s 
                    sum_Tp_work += t_fit_parallel_s
                    sum_T_total += t_fit_total_s 
                    
                    file.write(f"{i}, {t_fit_total_s:.6f}, {t_fit_serial_s:.6f}, {t_fit_parallel_s:.6f}\n")
                
                # --- Statistika ---
                avg_Ts_work = sum_Ts_work / NUM_RUNS
                avg_Tp_work = sum_Tp_work / NUM_RUNS
                avg_T_total = sum_T_total / NUM_RUNS
                total_ideal_work = avg_Ts_work + avg_Tp_work
                
                # Zapisivanje statistike
                file.write(f"\n=== Statistics (N={t}) ===\n")
                file.write(f"Average Wall-Clock Time: {avg_T_total:.4f} s\n")
                file.write(f"Average ideal work: {total_ideal_work:.4f} s\n")
                
            print(f"Completed N={t}. Results saved to {output_file_name}")

        except Exception as e:
            print(f"An error occurred during run with N={t}: {e}")
            
if __name__ == "__main__":
    main()