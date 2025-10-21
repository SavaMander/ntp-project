from __future__ import print_function
from datetime import timedelta
import time
import numpy, random, copy
import multiprocessing
import math

def _euclidean_distance(x, y):
    x_np = numpy.array(x)
    y_np = numpy.array(y)
    return numpy.linalg.norm(y_np - x_np)

def _assign_chunk(chunk_of_data, cluster_centers):
    assignments = []
    for point in chunk_of_data:
        min_distance = float('inf')
        cluster_index = -1
        
        for index, center in enumerate(cluster_centers):
            distance = _euclidean_distance(point, center)
            if distance < min_distance:
                min_distance = distance
                cluster_index = index
        
        assignments.append((point, cluster_index)) 
    
    return assignments

def _map_m_step(assignment_chunk, n_clusters, dimensions):
    # Initialize C results (sum_vector, count) for all clusters
    local_stats = []
    for _ in range(n_clusters):
        local_stats.append((numpy.zeros(dimensions).tolist(), 0))
    
    for point, cluster_index in assignment_chunk:
        current_sum, current_n = local_stats[cluster_index]
        
        # Accumulate sum
        point_np = numpy.array(point)
        current_sum_np = numpy.array(current_sum)
        new_sum_np = current_sum_np + point_np
        
        # Update local stats
        local_stats[cluster_index] = (new_sum_np.tolist(), current_n + 1)
        
    return local_stats

class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []

class KMeans(object):

    def __init__(self, n_clusters, max_iter, n_threads):
        self.n_threads = n_threads
        if n_threads > multiprocessing.cpu_count():
            self.n_threads = multiprocessing.cpu_count()
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []
        self.pool = None

    def fit(self, data):
            self.data = data
            total_serial_work = 0.0
            total_parallel_work = 0.0
            t_s_init_start = time.time()
            dimensions = len(self.data[0])
            log_filename = "results/parallel/results.txt"
            with open(log_filename, 'w') as f:
                f.write("")
            random.seed(123)
            numpy.random.seed(123)
            for i in range(self.n_clusters):
                point = [random.random() for x in range(dimensions)] 
                self.clusters.append(Cluster(point))
            t_s_init_end = time.time()
            total_serial_work += (t_s_init_end - t_s_init_start)
            iter_no = 0
            tolerance = 1e-4 
            n_processors = self.n_threads
            data_size = len(self.data)
            if n_processors > 1 and self.pool is None:
                self.pool = multiprocessing.Pool(processes=n_processors)
            
            chunk_size_e = int(math.ceil(data_size / n_processors)) 
            data_chunks = [self.data[i:i + chunk_size_e] for i in range(0, data_size, chunk_size_e)]
            
            while iter_no < self.max_iter:               
                # Parallel E-step (no change)
                t_e_start = time.time()
                current_centers = [c.center for c in self.clusters]
                args_e = [(chunk, current_centers) for chunk in data_chunks]
                if n_processors > 1:
                    results_e = self.pool.starmap(_assign_chunk, args_e)
                else:
                    results_e = [_assign_chunk(*args_e[0])] 
                t_e_end = time.time()
                total_parallel_work += (t_e_end - t_e_start)
                               
                # Updating centers (M-step)
                old_centers = [numpy.array(c.center) for c in self.clusters]
                
                # Map Phase (Parallel reduction using N threads)
                t_m_start = time.time()
                n_clusters = self.n_clusters
                
                args_m_opt = [(chunk, n_clusters, dimensions) for chunk in results_e]

                if n_processors > 1:
                    # N workers run _map_m_step_optimized in parallel
                    results_m_chunks = self.pool.starmap(_map_m_step, args_m_opt)
                else:
                    # Serial execution
                    results_m_chunks = [_map_m_step(results_e[0], n_clusters, dimensions)]
                    
                t_m_end = time.time()
                total_parallel_work += (t_m_end - t_m_start)
                
                # Reduce Phase (Serial global aggregation and center update)
                t_s3_start = time.time()
                
                # 1. Initialize final global stats for all C clusters
                final_stats = []
                for _ in range(n_clusters):
                    # Use numpy for final aggregation speed
                    final_stats.append((numpy.zeros(dimensions), 0))
                
                # 2. Global summation of partial results from all N workers
                for worker_stats in results_m_chunks:
                    for i in range(n_clusters):
                        partial_sum_list, partial_n = worker_stats[i]
                        
                        # Accumulate sum and count
                        final_stats[i] = (
                            final_stats[i][0] + numpy.array(partial_sum_list),
                            final_stats[i][1] + partial_n
                        )

                # 3. Final Center Calculation, Update, and Convergence Check
                moved_distance_sum = 0
                for i in range(n_clusters):
                    sum_of_data_np, n = final_stats[i]
                    
                    # Update center
                    if n > 0:
                        new_center = (sum_of_data_np / n).tolist()
                        self.clusters[i].center = new_center
                    
                    # Convergence check
                    new_center_np = numpy.array(self.clusters[i].center)
                    old_center_np = old_centers[i]
                    
                    distance = numpy.linalg.norm(new_center_np - old_center_np)
                    moved_distance_sum += distance
                    
                t_s3_end = time.time()
                total_serial_work += (t_s3_end - t_s3_start)
                
                with open(log_filename, 'a') as f:
                    f.write(f"ITERATION {iter_no}:\n")
                    # f.write(f"SSE: {sse_value:.4f}\n")
                    for i, cluster in enumerate(self.clusters):
                        center_coords = ", ".join(f"{coord:.4f}" for coord in cluster.center)
                        f.write(f"Center {i}: [{center_coords}]\n")
                    
                    f.write("---\n")
                
                if moved_distance_sum < tolerance:
                    break
                    
                iter_no += 1
                         
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
                
            total_time_s = total_serial_work + total_parallel_work
            return (
                timedelta(seconds=total_time_s),
                timedelta(seconds=total_serial_work),
                timedelta(seconds=total_parallel_work)
            )
    
    def predict(self, point):
        min_distance = None
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = _euclidean_distance(point, self.clusters[index].center) 
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance
        
        return cluster_index
        
    def sum_squared_error(self):
        sse = 0
        for cluster in self.clusters:
            for point in cluster.data:
                distance = _euclidean_distance(cluster.center, point)
                sse += distance**2 
        
        return sse
    