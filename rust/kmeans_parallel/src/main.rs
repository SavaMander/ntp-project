use kmeans_parallel::kmeans::{KMeans, StandardScaler, load_csv};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn Error>> {    
    let path = "../../data/movielens1m.csv";
    println!("Loading and Scaling data...");
    let mut data = load_csv(path, Some(10000))?;
    let mut scaler = StandardScaler::new();
    data = scaler.fit_transform(&data);

    let num_runs = 30;
    const N_THREADS: usize = 12;
    
    let mut sum_Ts_work = 0.0;
    let mut sum_Tp_work = 0.0;
    let mut sum_T_total = 0.0;

    println!("Running {} iterations for baseline (N={})...", num_runs, N_THREADS);
    
    for t in 1..=N_THREADS {
        let output_file_name = format!("strong_scaling/kmeans_n{}_results.csv", t);
        let mut file = File::create(output_file_name)?;
        writeln!(file, "Run,TotalTime_s,SerialTime_s,ParallelTime_s")?;
        sum_Ts_work = 0.0;
        sum_Tp_work = 0.0;
        sum_T_total = 0.0;
        for i in 1..=num_runs {
            let mut model = KMeans::new(3, 5, t);
            
            let (t_fit_total, t_fit_serial, t_fit_parallel) = model.fit(&data);
            let t_fit_total_s = t_fit_total.as_secs_f64();
            let t_fit_serial_s = t_fit_serial.as_secs_f64();
            let t_fit_parallel_s = t_fit_parallel.as_secs_f64();
            
            sum_Ts_work += t_fit_serial_s; 
            sum_Tp_work += t_fit_parallel_s;
            sum_T_total += t_fit_total_s; 
            
            writeln!(file, "{}, {:.6}, {:.6}, {:.6}", i, t_fit_total_s, t_fit_serial_s, t_fit_parallel_s)?;
        }
    
        let avg_Ts_work = sum_Ts_work / num_runs as f64;
        let avg_Tp_work = sum_Tp_work / num_runs as f64;
        let avg_T_total = sum_T_total / num_runs as f64;
        
        let total_ideal_work = avg_Ts_work + avg_Tp_work; // T_s_work + T_p_work
        
        writeln!(file, "\n=== Statistics (N={}) ===", t)?;
        writeln!(file, "Average Wall-Clock Time: {:.4} s", avg_T_total)?;
        writeln!(file,"Average ideal work: {:.4} s", total_ideal_work);
    }
    Ok(())
}
