use rand::Rng;
use std::error::Error;
use csv::ReaderBuilder;
use std::time::Instant;
use ndarray::{Array1, Array2, Axis};

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

    fn fit(&mut self, data: &Array2<f64>) {
        let start = Instant::now();
        let mut rng = rand::thread_rng();

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


fn main() -> Result<(), Box<dyn Error>> {
    let path = "../../data/movielens1m.csv";
    println!("Loading data...");
    let mut data = load_csv(path, None)?;
    println!("Scaling data...");
    let mut scaler = StandardScaler::new();
    data = scaler.fit_transform(&data);
    let start = Instant::now();
    println!("Running KMeans (sequential)...");
    let mut model = KMeans::new(3, 3);
    model.fit(&data);
    
    let duration = start.elapsed();
    println!("Execution time: {:.4} seconds", duration.as_secs_f64());
    println!("SSE: {:.4}", model.sum_squared_error());

    Ok(())
}

