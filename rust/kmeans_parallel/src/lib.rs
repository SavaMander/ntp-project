use rand::Rng;
use std::error::Error;
use csv::ReaderBuilder;
use std::thread;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use ndarray::{Array1, Array2, Axis, s};
use num_cpus;
use rand::SeedableRng; 
use rand_xorshift::XorShiftRng;
use linfa::DatasetBase;
use linfa::traits::Fit;
use linfa::prelude::Transformer;
use linfa_reduction::Pca;
use plotters::prelude::*;

pub struct StandardScaler {
    mean_: Option<Array1<f64>>,
    std_: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self { mean_: None, std_: None }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        self.mean_ = Some(mean);
        self.std_ = Some(std);
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean_.as_ref().expect("Scaler not fitted");
        let std = self.std_.as_ref().expect("Scaler not fitted");
        let eps = 1e-8;
        (data - mean) / (std + eps)
    }

    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

#[derive(Clone)]
pub struct Cluster {
    pub center: Array1<f64>,
    pub data: Vec<Array1<f64>>,
}

impl Cluster {
    pub fn new(center: Array1<f64>) -> Self {
        Cluster { center, data: Vec::new() }
    }
}

pub struct KMeans {
    n_clusters: usize,
    max_iter: usize,
    pub clusters: Vec<Cluster>,
    n_threads: usize
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iter: usize, n_threads: usize) -> Self {
        let mut threads = n_threads;
        if n_threads == 0 || n_threads > num_cpus::get() {
            threads = num_cpus::get();
        }
        KMeans { n_clusters, max_iter, clusters: Vec::new(), n_threads: threads }
    }

    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(std::time::Duration, std::time::Duration, std::time::Duration), Box<dyn Error>> {
        const FIXED_SEED: u64 = 42; 
        let mut rng = XorShiftRng::seed_from_u64(FIXED_SEED);
        let start_init = Instant::now();
        let log_filename = "result.txt";
        let mut log_file = File::create(log_filename)?;
        
        // Initialize cluster centers
        for _ in 0..self.n_clusters {
            let random_index = rng.gen_range(0..data.shape()[0]);
            let point = data.row(random_index).to_owned();
            self.clusters.push(Cluster::new(Array1::from(point)));
        }
        let time_init = start_init.elapsed();
        
        let tolerance = 1e-4;

        let mut total_serial_time = std::time::Duration::new(0, 0);
        let mut total_parallel_time = std::time::Duration::new(0, 0);
        let mut total_iter_time = std::time::Duration::new(0, 0);
        
        for iter in 0..self.max_iter {
            // Centers for E-step.
            let iter_start = Instant::now();
            let t_serial_start = Instant::now();
            let centers: Vec<Array1<f64>> = self.clusters.iter().map(|c| c.center.clone()).collect();
            
            // Clear data in clusters for new E-step
            for cluster in &mut self.clusters {
                cluster.data.clear();
            }
            let t_serial_end = Instant::now();
            
            // Parallel E-step
            let t_e_start = Instant::now();
            let chunk_size_e = (data.nrows() + self.n_threads - 1) / self.n_threads;
            let mut handles = Vec::new();

            for chunk_start in (0..data.nrows()).step_by(chunk_size_e) {
                let chunk_end = usize::min(chunk_start + chunk_size_e, data.nrows());
                let centers_clone = centers.clone();
                let data_chunk = data.slice(s![chunk_start..chunk_end, ..]).to_owned();

                handles.push(thread::spawn(move || {
                    let mut local_assignments = Vec::new();
                    for (i, row) in data_chunk.rows().into_iter().enumerate() {
                        let idx = closest_cluster(&row.to_owned(), &centers_clone);
                        local_assignments.push((chunk_start + i, idx));
                    }
                    local_assignments
                }));
            }

            // Collect results
            let mut all_assignments = Vec::new();
            for handle in handles {
                let mut result = handle.join().unwrap();
                all_assignments.append(&mut result);
            }
            let t_e_end = Instant::now();
            
            // Assign data to clusters
            let t_serial2_start = Instant::now();
            for (row_index, cluster_index) in all_assignments {
                self.clusters[cluster_index].data.push(data.row(row_index).to_owned());
            }
            let t_serial2_end = Instant::now();

            // PARALLEL M-step
            let t_m_start = Instant::now();
            let old_centers: Vec<Array1<f64>> = self.clusters.iter().map(|c| c.center.clone()).collect();
            let n_clusters = self.n_clusters;
            let n_dims = data.ncols();
            
            // Clone point vectors for threads
            let cluster_data_copies: Vec<Vec<Array1<f64>>> = self.clusters.iter()
                .map(|c| c.data.clone())
                .collect();
            
            // 1. Preparing for Map-Reduce: Aggregation of results for all clusters
            let mut total_sums: Vec<Array1<f64>> = (0..n_clusters).map(|_| Array1::zeros(n_dims)).collect();
            let mut total_counts: Vec<f64> = vec![0.0; n_clusters];

            // Split clusters to threads
            let chunk_size_m = (n_clusters + self.n_threads - 1) / self.n_threads; 
            let mut handles = Vec::new();

            for thread_id in 0..self.n_threads {
                let start_cluster_idx = thread_id * chunk_size_m;

                if start_cluster_idx >= n_clusters {
                    break; 
                }
                
                let end_cluster_idx = usize::min(start_cluster_idx + chunk_size_m, n_clusters);
                
                let local_cluster_data_refs: Vec<Vec<Array1<f64>>> = 
                    cluster_data_copies[start_cluster_idx..end_cluster_idx].to_vec();

                handles.push(thread::spawn(move || {
                    let num_local_clusters = end_cluster_idx - start_cluster_idx;
                    let mut local_sums: Vec<Array1<f64>> = (0..num_local_clusters).map(|_| Array1::zeros(n_dims)).collect();
                    let mut local_counts: Vec<f64> = vec![0.0; num_local_clusters];

                    // MAP: Summing the coordinates and counting data for this cluster
                    for (local_idx, data_points) in local_cluster_data_refs.iter().enumerate() {
                        let mut sum = Array1::<f64>::zeros(n_dims);
                        for point in data_points.iter() {
                            sum = &sum + point;
                        }
                        local_sums[local_idx] = sum;
                        local_counts[local_idx] = data_points.len() as f64;
                    }
                    (start_cluster_idx, local_sums, local_counts) 
                }));
            }
            
            // 2. Reduce/Aggregate
            for handle in handles {
                let (start_idx, local_sums, local_counts) = handle.join().unwrap();
                for i in 0..(local_sums.len()) {
                    let global_idx = start_idx + i;
                    total_sums[global_idx] = &total_sums[global_idx] + &local_sums[i];
                    total_counts[global_idx] += local_counts[i];
                }
            }

            // 3. Final center: Finding new centers and updating self.clusters
            for i in 0..n_clusters {
                let sum = &total_sums[i];
                let count = total_counts[i];
                if count > 0.0 {
                    self.clusters[i].center = sum / count;
                }
            }
            let t_m_end = Instant::now();
            let iter_end = Instant::now();

            writeln!(log_file, "ITERATION {}:", iter)?;

            for (i, cluster) in self.clusters.iter().enumerate() {
                let center_coords = cluster.center.iter()
                    .map(|&c| format!("{:.4}", c))
                    .collect::<Vec<String>>()
                    .join(", ");
                writeln!(log_file, "Center {}: [{}]", i, center_coords)?;
            }
            writeln!(log_file, "---")?;
            
            // Convergence check
            let t_serial3_start = Instant::now();
            let moved_distance_sum: f64 = old_centers.iter().zip(&self.clusters)
                .map(|(old, new)| euclidean_distance(old, &new.center))
                .sum();
            let t_serial3_end = Instant::now();

            let serial_time = (t_serial_end - t_serial_start)
                + (t_serial2_end - t_serial2_start)
                + (t_serial3_end - t_serial3_start);
            let e_time = t_e_end - t_e_start;
            let m_time = t_m_end - t_m_start;
            let total_time = iter_end - iter_start;
            let parallel_time = e_time + m_time;

            total_serial_time += serial_time;
            total_parallel_time += parallel_time;
            total_iter_time += total_time;

            if moved_distance_sum < tolerance {
                break;
            }
        }

        let final_ts = time_init + total_serial_time;
        Ok((time_init + total_iter_time, final_ts, total_parallel_time))
    }

    pub fn sum_squared_error(&self) -> f64 {
        let mut sse = 0.0;
        for cluster in &self.clusters {
            for point in &cluster.data {
                let d = euclidean_distance(&cluster.center, point);
                sse += d.powi(2);
            }
        }
        sse
    }
}

fn closest_cluster(point: &Array1<f64>, centers: &[Array1<f64>]) -> usize {
    let mut min_dist = f64::INFINITY;
    let mut idx = 0;
    for (i, center) in centers.iter().enumerate() {
        let d = euclidean_distance(point, center);
        if d < min_dist {
            min_dist = d;
            idx = i;
        }
    }
    idx
}

fn euclidean_distance(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    (y - x).mapv(|v| v.powi(2)).sum().sqrt()
}

pub fn load_csv(path: &str, n_rows_limit: Option<usize>) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut records = Vec::new();

    for (i, result) in reader.records().enumerate() {
        if let Some(limit) = n_rows_limit {
            if i >= limit { break; }
        }
        let record = result?;
        let row: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap_or(0.0)).collect();
        records.push(row);
    }

    let n_rows = records.len();
    let n_cols = if n_rows > 0 { records[0].len() } else { 0 };
    let flat: Vec<f64> = records.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n_rows, n_cols), flat)?)
}

pub struct CenterLog {
    pub iteration: usize,
    pub centers: Vec<Array1<f64>>,
}

pub fn load_center_log(log_filename: &str, _dimensions: usize) -> Result<Vec<CenterLog>, Box<dyn Error>> {
    let file = File::open(log_filename)?;
    let reader = BufReader::new(file);
    let mut logs = Vec::new();
    let mut current_iteration = None;
    let mut current_centers = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("ITERATION") {
            if let Some(iter) = current_iteration {
                logs.push(CenterLog { iteration: iter, centers: current_centers.drain(..).collect() });
            }
            let iter_str = line.split(" ").nth(1).ok_or("Invalid ITERATION format")?;
            current_iteration = Some(usize::from_str(iter_str.trim_end_matches(':'))?);
        } else if line.starts_with("Center") {
            let coords_start = line.find('[').ok_or("Invalid center format")? + 1;
            let coords_end = line.find(']').ok_or("Invalid center format")?;
            let coords_str = &line[coords_start..coords_end];
            let coords: Vec<f64> = coords_str.split(", ")
                .map(|s| f64::from_str(s).unwrap_or(0.0))
                .collect();
            current_centers.push(Array1::from(coords));
        }
    }

    if let Some(iter) = current_iteration {
        logs.push(CenterLog { iteration: iter, centers: current_centers });
    }

    Ok(logs)
}

pub fn visualize_kmeans_pca(
    original_data: &Array2<f64>, 
    center_logs: Vec<CenterLog>, 
    n_clusters: usize
) -> Result<(), Box<dyn Error>> {
    println!("Calculating PCA for visualization...");

    let dataset = DatasetBase::new(original_data.view(), ());
    let pca = Pca::params(2).fit(&dataset)?;
    
    let data_dataset = DatasetBase::new(original_data.view(), ());
    let projected_data = pca.transform(data_dataset);
    let projected_array = projected_data.records();

    let projected_points: Vec<(f64, f64)> = projected_array.rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();

    let x_min = projected_array.column(0).fold(f64::MAX, |acc, &x| acc.min(x)) - 0.5;
    let x_max = projected_array.column(0).fold(f64::MIN, |acc, &x| acc.max(x)) + 0.5;
    let y_min = projected_array.column(1).fold(f64::MAX, |acc, &y| acc.min(y)) - 0.5;
    let y_max = projected_array.column(1).fold(f64::MIN, |acc, &y| acc.max(y)) + 0.5;

    let cluster_colors: Vec<RGBAColor> = vec![
        RED.mix(0.8), GREEN.mix(0.8), BLUE.mix(0.8), 
        MAGENTA.mix(0.8), YELLOW.mix(0.8), CYAN.mix(0.8)
    ].into_iter().take(n_clusters).collect();

    for log in center_logs {
        let iter = log.iteration;
        let output_path = format!("visualization/kmeans_iter_{}.png", iter);

        let n_centers = log.centers.len();
        let n_dims = log.centers[0].len();
        let center_matrix = Array2::from_shape_vec(
            (n_centers, n_dims),
            log.centers.iter().flat_map(|a| a.to_vec()).collect()
        )?;

        let centers_dataset = DatasetBase::new(center_matrix.view(), ());
        let projected_centers = pca.transform(centers_dataset);
        let projected_centers_array = projected_centers.records();
        
        let current_projected_centers: Vec<(f64, f64)> = projected_centers_array.rows()
            .into_iter()
            .map(|row| (row[0], row[1]))
            .collect();

        let root = BitMapBackend::new(&output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("K-Means Iteracija {} (PCA Projekcija)", iter), ("sans-serif", 40).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(
            projected_points.iter().map(|(x, y)| Circle::new((*x, *y), 2, BLACK.mix(0.1).filled()))
        )?;

        for (i, &center) in current_projected_centers.iter().enumerate() {
            let color = cluster_colors[i % cluster_colors.len()];

            chart.draw_series(std::iter::once(Circle::new(center, 8, color.filled())))?
                .label(format!("Centar {}", i))
                .legend(move |(x, y)| Circle::new((x, y), 5, color.filled()));
        }

        chart.configure_series_labels().border_style(&BLACK).draw()?;
        root.present()?;
    }

    Ok(())
}