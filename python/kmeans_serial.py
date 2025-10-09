from __future__ import print_function
import numpy, random, copy


class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []  # data that belongs to this cluster

    def recalculate_center(self):
            if not self.data:
                return 
            
            data_np = numpy.array(self.data)
            
            # Calculating sum of points by coordinates (sum of columns)
            sum_of_data = numpy.sum(data_np, axis=0)
            
            # Number of data in cluster
            n = len(self.data)
            
            # New center
            self.center = (sum_of_data / n).tolist()
                
class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data):
        self.data = data
        
        # Dimensions of space
        dimensions = len(self.data[0])
        # N random points as initial cluster centers
        for i in range(self.n_clusters):
            point = [random.random() for x in range(dimensions)]
            self.clusters.append(Cluster(point))
        
        iter_no = 0
        tolerance = 1e-4  # small tolerance for convergence check
        
        while iter_no < self.max_iter: # Check < max_iter
            # Cleaning the cluster
            for cluster in self.clusters:
                cluster.data = []
                
            # Cluster assignment (E-step)
            for point in self.data:
                cluster_index = self.predict(point)
                self.clusters[cluster_index].data.append(point)
                
            # Update centers (M-step)
            # Keeping old centers
            old_centers = [numpy.array(c.center) for c in self.clusters]
            
            # Calculating new centers
            for cluster in self.clusters:
                cluster.recalculate_center()
            
            # Convergence check
            moved_distance_sum = 0
            for i in range(self.n_clusters):
                new_center = numpy.array(self.clusters[i].center)
                old_center = old_centers[i]
                
                # Calculating distance between new and old center
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
            distance = self.euclidean_distance(point, self.clusters[index].center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance
        
        return cluster_index
    
    def euclidean_distance(self, x, y):
        x_np = numpy.array(x)
        y_np = numpy.array(y)
        return numpy.linalg.norm(y_np - x_np)
        
    def sum_squared_error(self):
        sse = 0
        for cluster in self.clusters:
            for point in cluster.data:
                distance = self.euclidean_distance(cluster.center, point)
                sse += distance**2 
        
        return sse
