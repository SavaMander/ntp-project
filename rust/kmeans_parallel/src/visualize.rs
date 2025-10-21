use kmeans_parallel::{KMeans, StandardScaler, load_csv, load_center_log, visualize_kmeans_pca};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let log_file = "result.txt";
    let n_clusters = 5;
    let path = "../../data/movielens1m.csv";

    println!("Loading data...");
    let mut data = load_csv(path, Some(10000))?;

    println!("Scaling data...");
    let mut scaler = StandardScaler::new();
    data = scaler.fit_transform(&data);

    let start = Instant::now();

    println!("Running KMeans (sequential)...");
    let mut model = KMeans::new(n_clusters, 100,12);
    model.fit(&data)?;

    std::fs::create_dir_all("visualization")?;

    // Load log file with cluster centers
    let center_logs = load_center_log(log_file, data.shape()[1])?;

    // Visualize KMeans progress with PCA
    visualize_kmeans_pca(&data, center_logs, n_clusters)?;

    let duration = start.elapsed();
    println!("Execution time: {:.4} seconds", duration.as_secs_f64());
    println!("SSE: {:.4}", model.sum_squared_error());

    Ok(())
}