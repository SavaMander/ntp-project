use rand::Rng;
use std::error::Error;
use csv::ReaderBuilder;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use ndarray::{Array1, Array2, Axis};
use linfa_reduction::Pca;
use linfa::traits::Fit;
use plotters::prelude::*;
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use linfa::DatasetBase;
use linfa::prelude::Transformer;

struct StandardScaler {
    mean_: Option<Array1<f64>>,
    std_: Option<Array1<f64>>,
}

impl StandardScaler {
    fn new() -> Self {
        Self {
            mean_: None,
            std_: None,
        }
    }

    fn fit(&mut self, data: &Array2<f64>) {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        self.mean_ = Some(mean);
        self.std_ = Some(std);
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean_.as_ref().expect("Scaler not fitted");
        let std = self.std_.as_ref().expect("Scaler not fitted");
        let eps = 1e-8;
        (data - mean) / (std + eps)
    }

    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

#[derive(Clone)]
struct Cluster {
    center: Array1<f64>,
    data: Vec<Array1<f64>>,
}

impl Cluster {
    fn new(center: Array1<f64>) -> Self {
        Cluster { center, data: Vec::new() }
    }

    fn recalculate_center(&mut self) {
        if self.data.is_empty() {
            return;
        }
        let n_points = self.data.len() as f64;
        let mut sum = Array1::<f64>::zeros(self.center.len());

        for point in &self.data {
            sum = &sum + point;
        }

        self.center = &sum / n_points;
    }
}

struct KMeans {
    n_clusters: usize,
    max_iter: usize,
    clusters: Vec<Cluster>,
}

impl KMeans {
    fn new(n_clusters: usize, max_iter: usize) -> Self {
        KMeans {
            n_clusters,
            max_iter,
            clusters: Vec::new(),
        }
    }

    fn fit(&mut self, data: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        let start = Instant::now();
        let mut rng = rand::thread_rng();
        let log_filename = "result.txt";
        let mut log_file = File::create(log_filename)?;

        for _ in 0..self.n_clusters {
            let random_index = rng.gen_range(0..data.shape()[0]);
            let point = data.row(random_index).to_owned();
            self.clusters.push(Cluster::new(Array1::from(point)));
        }

        let tolerance = 1e-4;
        for iter in 0..self.max_iter {
            for cluster in &mut self.clusters {
                cluster.data.clear();
            }

            for row in data.rows() {
                let cluster_index = self.predict(&row.to_owned());
                self.clusters[cluster_index].data.push(row.to_owned());
            }

            let old_centers: Vec<Array1<f64>> = self.clusters.iter().map(|c| c.center.clone()).collect();

            for cluster in &mut self.clusters {
                cluster.recalculate_center();
            }

            writeln!(log_file, "ITERATION {}:", iter)?;

            for (i, cluster) in self.clusters.iter().enumerate() {
                let center_coords = cluster.center.iter()
                    .map(|&c| format!("{:.4}", c))
                    .collect::<Vec<String>>()
                    .join(", ");
                writeln!(log_file, "Center {}: [{}]", i, center_coords)?;
            }
            writeln!(log_file, "---")?;

            let mut moved_distance_sum = 0.0;
            for (old, new_cluster) in old_centers.iter().zip(&self.clusters) {
                moved_distance_sum += euclidean_distance(old, &new_cluster.center);
            }

            let duration = start.elapsed();
            println!(
                "Iteration {}: total {:.3?}",
                iter + 1,
                duration,
            );

            if moved_distance_sum < tolerance {
                break;
            }
        }
        Ok(())
    }

    fn predict(&self, point: &Array1<f64>) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut cluster_index = 0;

        for (i, cluster) in self.clusters.iter().enumerate() {
            let distance = euclidean_distance(point, &cluster.center);
            if distance < min_distance {
                min_distance = distance;
                cluster_index = i;
            }
        }

        cluster_index
    }

    fn sum_squared_error(&self) -> f64 {
        let mut sse = 0.0;
        for cluster in &self.clusters {
            for point in &cluster.data {
                let distance = euclidean_distance(&cluster.center, point);
                sse += distance.powi(2);
            }
        }
        sse
    }
}

fn euclidean_distance(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    (y - x).mapv(|v| v.powi(2)).sum().sqrt()
}

fn load_csv(path: &str, n_rows_limit: Option<usize>) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut records = Vec::new();

    for (i, result) in reader.records().enumerate() {
        if let Some(limit) = n_rows_limit {
            if i >= limit {
                break;
            }
        }
        let record = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();
        records.push(row);
    }

    let n_rows = records.len();
    let n_cols = if n_rows > 0 { records[0].len() } else { 0 };
    let flat: Vec<f64> = records.into_iter().flatten().collect();

    Ok(Array2::from_shape_vec((n_rows, n_cols), flat)?)
}

struct CenterLog {
    iteration: usize,
    centers: Vec<Array1<f64>>,
}

fn load_center_log(log_filename: &str, _dimensions: usize) -> Result<Vec<CenterLog>, Box<dyn Error>> {
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

fn visualize_kmeans_pca(
    original_data: &Array2<f64>, 
    center_logs: Vec<CenterLog>, 
    n_clusters: usize
) -> Result<(), Box<dyn Error>> {
    println!("Calculating PCA for visualization...");

    // Create dataset and fit PCA
    let dataset = DatasetBase::new(original_data.view(), ());
    let pca = Pca::params(2).fit(&dataset)?;
    
    // Transform data - wrap in DatasetBase
    let data_dataset = DatasetBase::new(original_data.view(), ());
    let projected_data = pca.transform(data_dataset);

    // Extract the array from the DatasetBase
    let projected_array = projected_data.records();

    // Convert projected data to points
    let projected_points: Vec<(f64, f64)> = projected_array.rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();

    // Determine chart ranges
    let x_min = projected_array.column(0).fold(f64::MAX, |acc, &x| acc.min(x)) - 0.5;
    let x_max = projected_array.column(0).fold(f64::MIN, |acc, &x| acc.max(x)) + 0.5;
    let y_min = projected_array.column(1).fold(f64::MAX, |acc, &y| acc.min(y)) - 0.5;
    let y_max = projected_array.column(1).fold(f64::MIN, |acc, &y| acc.max(y)) + 0.5;

    let cluster_colors: Vec<RGBAColor> = vec![
        RED.mix(0.8), GREEN.mix(0.8), BLUE.mix(0.8), 
        MAGENTA.mix(0.8), YELLOW.mix(0.8), CYAN.mix(0.8)
    ].into_iter().take(n_clusters).collect();

    let mut previous_projected_centers: Option<Vec<(f64, f64)>> = None;

    for log in center_logs {
        let iter = log.iteration;
        let output_path = format!("visualization/kmeans_iter_{}.png", iter);

        // Convert centers into Array2
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

        // Draw chart
        let root = BitMapBackend::new(&output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("K-Means Iteracija {} (PCA Projekcija)", iter), ("sans-serif", 40).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.configure_mesh().draw()?;

        // Draw data points
        chart.draw_series(
            projected_points.iter().map(|(x, y)| Circle::new((*x, *y), 2, BLACK.mix(0.1).filled()))
        )?;

        // Draw centers and movement
        for (i, &center) in current_projected_centers.iter().enumerate() {
            let color = cluster_colors[i % cluster_colors.len()];

            // Center point
            chart.draw_series(std::iter::once(Circle::new(center, 8, color.filled())))?
                .label(format!("Centar {}", i))
                .legend(move |(x, y)| Circle::new((x, y), 5, color.filled()));
        }

        chart.configure_series_labels().border_style(&BLACK).draw()?;
        root.present()?;

        previous_projected_centers = Some(current_projected_centers);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let log_file = "result.txt";
    let n_clusters = 10;
    let path = "../../data/movielens1m.csv";

    println!("Loading data...");
    let mut data = load_csv(path, Some(1000))?;

    println!("Scaling data...");
    let mut scaler = StandardScaler::new();
    data = scaler.fit_transform(&data);

    let start = Instant::now();

    println!("Running KMeans (sequential)...");
    let mut model = KMeans::new(n_clusters, 100);
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