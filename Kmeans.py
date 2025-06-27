__authors__ = []
__group__ = '80'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        if self.K > 0:
            self._init_centroids()
            self.labels = np.zeros(len(self.X))

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if len(X.shape) == 3:  # Pass from 3d to 2d
            self.X = X.reshape((X.shape[0] * X.shape[1], X.shape[2])).astype(float)
        else:
            self.X = X.astype(float)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.zeros((self.K, self.X.shape[1]))
            
            if len(self.X) > 10000:
                unique_points, unique_indices = np.unique(self.X, axis=0, return_index=True)
                k_points = min(self.K, len(unique_points))
                self.centroids[:k_points] = unique_points[:k_points]
                
                if k_points < self.K:
                    min_vals = np.min(self.X, axis=0)
                    max_vals = np.max(self.X, axis=0)
                    self.centroids[k_points:] = np.random.uniform(
                        low=min_vals, high=max_vals, size=(self.K - k_points, self.X.shape[1])
                    )
            else:
                used_points = set()
                centroid_count = 0
                
                for idx, point in enumerate(self.X):
                    point_tuple = tuple(point)
                    
                    if point_tuple not in used_points:
                        self.centroids[centroid_count] = point
                        used_points.add(point_tuple)
                        centroid_count += 1
                        
                        if centroid_count >= self.K:
                            break
            
            self.old_centroids = self.centroids.copy() + 0.1
        else:
            min_vals = np.min(self.X, axis=0)
            max_vals = np.max(self.X, axis=0)
            
            self.centroids = np.random.uniform(low=min_vals, high=max_vals, size=(self.K, self.X.shape[1]))
            self.old_centroids = np.random.uniform(low=min_vals, high=max_vals, size=(self.K, self.X.shape[1]))
        
        self.centroids = self.centroids.astype(float)
        self.old_centroids = self.old_centroids.astype(float)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        dist_matrix = distance(self.X, self.centroids)
        self.labels = np.argmin(dist_matrix, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()
        self.centroids = np.zeros_like(self.old_centroids)
        
        for k in range(self.K):
            mask = (self.labels == k)
            if np.any(mask):
                self.centroids[k] = np.mean(self.X[mask], axis=0)
            else:
                self.centroids[k] = self.old_centroids[k]

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        centroid_distances = np.sqrt(np.sum((self.centroids - self.old_centroids) ** 2, axis=1))
        return np.all(centroid_distances <= self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self.num_iter = 0
        self.get_labels()
        
        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_centroids()
            self.get_labels()
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        total_distance = 0
        total_points = len(self.X)
        
        assigned_centroids = self.centroids[self.labels]
        squared_dists = np.sum((self.X - assigned_centroids) ** 2, axis=1)
        total_distance = np.sum(squared_dists)
            
        return total_distance / total_points if total_points > 0 else 0

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        wcd_list = []

        for k in range(2, max_K + 1):
            self.K = k
            self._init_centroids()
            self.fit()
            wcd = self.withinClassDistance()
            wcd_list.append(wcd)

        percentage_decrease = []
        for i in range(1, len(wcd_list)):
            decrease = 100 * (wcd_list[i - 1] - wcd_list[i]) / wcd_list[i - 1]
            percentage_decrease.append(decrease)

        threshold = 20
        best_K = max_K
        for i, perc_dec in enumerate(percentage_decrease, start=2):
            if perc_dec < threshold:
                best_K = i
                break
        self.K = best_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    X_expanded = X[:, np.newaxis, :]  # Shape: (P,1,D)
    squared_diff = (X_expanded - C[np.newaxis, :, :]) ** 2  # Shape: (P,K,D)
    dist_matrix = np.sqrt(np.sum(squared_diff, axis=2))  # Shape: (P,K)
    
    return dist_matrix


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    labels = []
    prob = utils.get_color_prob(centroids)
    
    orange_idx = 1  # Orange
    pink_idx = 7    # Pink
    grey_idx = 9    # Grey
    
    for i in range(prob.shape[0]):
        centroid_probs = prob[i]
        
        max_prob = np.max(centroid_probs)
        threshold = 0.3 * max_prob
        
        if centroid_probs[orange_idx] > threshold and centroid_probs[orange_idx] > 0.1:
            labels.append(utils.colors[orange_idx])
        elif centroid_probs[pink_idx] > threshold and centroid_probs[pink_idx] > 0.1:
            labels.append(utils.colors[pink_idx])
        elif centroid_probs[grey_idx] > threshold and centroid_probs[grey_idx] > 0.1:
            labels.append(utils.colors[grey_idx])
        else:
            max_idx = np.argmax(centroid_probs)
            labels.append(utils.colors[max_idx])

    return labels
