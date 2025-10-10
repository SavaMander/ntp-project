use kmeans_parallel::kmeans::{KMeans, StandardScaler, load_csv};

use std::fs::File;
use std::io::Write;
use std::time::Instant;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../../data/movielens1m.csv";
    println!("Loading full dataset...");
    let full_data = load_csv(path, None)?; // load entire dataset
    let mut scaler = StandardScaler::new();
    let full_data = scaler.fit_transform(&full_data);

    const NUM_RUNS: usize = 30;
    const MAX_THREADS: usize = 12;
    const BASE_ROWS_PER_THREAD: usize = 10000; // Each thread handles ~1000 points

    std::fs::create_dir_all("weak_scaling")?;

    for n_threads in 1..=MAX_THREADS {
        let num_rows = BASE_ROWS_PER_THREAD * n_threads;
        if num_rows > full_data.nrows() {
            println!("⚠️ Not enough rows for {} threads, skipping...", n_threads);
            break;
        }

        let data_slice = full_data.slice(ndarray::s![0..num_rows, ..]).to_owned();
        let output_file_name = format!("weak_scaling/kmeans_n{}_results.csv", n_threads);
        let mut file = File::create(&output_file_name)?;
        writeln!(file, "Run,TotalTime_s,SerialTime_s,ParallelTime_s")?;

        let mut sum_T_total = 0.0;
        let mut sum_T_serial = 0.0;
        let mut sum_T_parallel = 0.0;

        println!("▶ Threads: {}, Data size: {}", n_threads, num_rows);

        for run in 1..=NUM_RUNS {
            let mut model = KMeans::new(3, 5, n_threads);

            let (t_total, t_serial, t_parallel) = model.fit(&data_slice);

            let t_total_s = t_total.as_secs_f64();
            let t_serial_s = t_serial.as_secs_f64();
            let t_parallel_s = t_parallel.as_secs_f64();

            sum_T_total += t_total_s;
            sum_T_serial += t_serial_s;
            sum_T_parallel += t_parallel_s;

            writeln!(file, "{}, {:.6}, {:.6}, {:.6}", run, t_total_s, t_serial_s, t_parallel_s)?;
        }

        let avg_total = sum_T_total / NUM_RUNS as f64;
        let avg_serial = sum_T_serial / NUM_RUNS as f64;
        let avg_parallel = sum_T_parallel / NUM_RUNS as f64;

        writeln!(file, "\n=== Statistics (Threads={}) ===", n_threads)?;
        writeln!(file, "Average Wall-Clock Time: {:.4} s", avg_total)?;
        writeln!(file, "Average ideal work: {:.4} s", avg_serial + avg_parallel)?;
    }

    Ok(())
}
