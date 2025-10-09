from __future__ import print_function
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

def _map_m_step(cluster_data_list, dimensions):
    results = []
    
    for data_points in cluster_data_list:
        if not data_points:
            sum_of_data = numpy.zeros(dimensions).tolist()
            n = 0
        else:
            data_np = numpy.array(data_points)
            sum_of_data = numpy.sum(data_np, axis=0).tolist() 
            n = len(data_points)
            
        results.append((sum_of_data, n))
        
    return results

class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []

class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data):
        self.data = data
        
        dimensions = len(self.data[0])
        for i in range(self.n_clusters):
            point = [random.random() for x in range(dimensions)] 
            self.clusters.append(Cluster(point))
        
        iter_no = 0
        tolerance = 1e-4 
        n_processors = multiprocessing.cpu_count()
        data_size = len(self.data)
        
        while iter_no < self.max_iter:
            # Clearing cluster data for new E-step
            for cluster in self.clusters:
                cluster.data = []
            
            # Parallel E-step
            chunk_size_e = int(math.ceil(data_size / n_processors)) 
            data_chunks = [self.data[i:i + chunk_size_e] for i in range(0, data_size, chunk_size_e)]
            current_centers = [c.center for c in self.clusters]
            args_e = [(chunk, current_centers) for chunk in data_chunks]
            
            with multiprocessing.Pool(processes=n_processors) as pool:
                results_e = pool.starmap(_assign_chunk, args_e)
                
            # Assign data to clusters
            for result_chunk in results_e:
                for point, cluster_index in result_chunk:
                    self.clusters[cluster_index].data.append(point)
                
            # Updating centers (M-step)
            old_centers = [numpy.array(c.center) for c in self.clusters]
            
            # Parallel M-step (Map-Reduce)
            
            # 1. Prepare data for Map phase
            cluster_data_copies = [c.data for c in self.clusters]
            n_clusters = self.n_clusters
            
            # 2. Split clusters for threads
            chunk_size_m = int(math.ceil(n_clusters / n_processors))
            
            # Creating chunks of clusters for workers
            cluster_chunks = [cluster_data_copies[i:i + chunk_size_m] 
                              for i in range(0, n_clusters, chunk_size_m)]

            args_m = [(chunk, dimensions) for chunk in cluster_chunks]

            all_stats = []
            with multiprocessing.Pool(processes=n_processors) as pool:
                # Map phase: Parallel sum and counting points
                results_m = pool.starmap(_map_m_step, args_m)
                
                # Reduce phase: Aggregation of results
                for chunk_stats in results_m:
                    # chunk_stats list [(sum_1, count_1), (sum_2, count_2), ...]
                    all_stats.extend(chunk_stats)

            # 3. Final center: Update self.clusters
            for i in range(n_clusters):
                sum_of_data_list, n = all_stats[i]
                
                if n > 0:
                    sum_of_data_np = numpy.array(sum_of_data_list)
                    # New center
                    new_center = (sum_of_data_np / n).tolist()
                    self.clusters[i].center = new_center
            
            # Convergence check
            moved_distance_sum = 0
            for i in range(self.n_clusters):
                new_center = numpy.array(self.clusters[i].center)
                old_center = old_centers[i]
                
                distance = numpy.linalg.norm(new_center - old_center)
                moved_distance_sum += distance
                
            #print("Iter no: " + str(iter_no) + ", Change of center: " + str(moved_distance_sum))
            
            if moved_distance_sum < tolerance:
                break
                
            iter_no += 1

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