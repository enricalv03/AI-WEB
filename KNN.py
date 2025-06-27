__authors__ = []
__group__ = '80'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold


class KNN:
    def __init__(self, train_data, labels, metric='euclidean'):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.metric = metric
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = train_data.astype(np.float32)
        P, M, N = train_data.shape
        self.train_data = train_data.reshape(P, M * N)
        
    def _calculate_distance(self, test_data, metric=None):
        """
        Calculates distance between test data and training data points
        :param test_data: test data reshaped to NxD
        :param metric: distance metric ('euclidean', 'manhattan', 'minkowski')
        :return: distance matrix
        """
        metric = metric or self.metric
        p = 3  # p value for Minkowski distance
        
        if metric == 'manhattan':
            return cdist(test_data, self.train_data, metric='cityblock')
        elif metric == 'minkowski':
            return cdist(test_data, self.train_data, metric='minkowski', p=p)
        else:  # default to euclidean
            return cdist(test_data, self.train_data, metric='euclidean')

    def get_k_neighbours(self, test_data, k, metric=None):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :param metric: distance metric to use
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        P, M, N = test_data.shape
        test_data = test_data.reshape(P, M * N)

        distances = self._calculate_distance(test_data, metric)

        sorted_indices = np.argsort(distances, axis=1)
        nearest_indices = np.zeros((distances.shape[0], k), dtype=int)
        for i in range(distances.shape[0]):
            nearest_indices[i] = sorted_indices[i][:k]

        neighbors_list = []

        for row in nearest_indices:
            neighbors_list.append([self.labels[idx] for idx in row])
        self.neighbors = np.array(neighbors_list)

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        classes = []
        for i in self.neighbors:
            counts = {}
            for j in i:
                counts[j] = counts.get(j, 0) + 1
            max_count = max(counts.values())
            for k in i:
                if counts[k] == max_count:
                    classes.append(k)
                    break
        return np.array(classes)

    def predict(self, test_data, k, metric=None):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :param metric: distance metric to use
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k, metric)
        return self.get_class()
        
    def predict_with_confidence(self, test_data, k, metric=None):
        """
        predicts the class with confidence score
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :param metric: distance metric to use
        :return: predicted classes and confidence scores
        """
        self.get_k_neighbours(test_data, k, metric)
        classes = []
        confidences = []
        
        for neighbors_row in self.neighbors:
            # Count occurrences of each class
            counts = {}
            for label in neighbors_row:
                counts[label] = counts.get(label, 0) + 1
                
            # Find class with maximum votes
            max_count = max(counts.values())
            for label in neighbors_row:
                if counts[label] == max_count:
                    predicted_class = label
                    break
                    
            # Calculate confidence as ratio of votes to k
            confidence = max_count / k
            classes.append(predicted_class)
            confidences.append(confidence)
            
        return np.array(classes), np.array(confidences)
    
    def find_best_metric(self, test_data, test_labels, k=5, n_folds=5):
        """
        Find the best distance metric using cross-validation
        :param test_data: validation data
        :param test_labels: validation labels
        :param k: number of neighbors
        :param n_folds: number of cross-validation folds
        :return: best metric and corresponding accuracy
        """
        metrics = ['euclidean', 'manhattan', 'minkowski']
        best_metric = None
        best_accuracy = -1
        
        # Combine training and validation data for cross-validation
        all_data = np.vstack([self.train_data, test_data.reshape(test_data.shape[0], -1)])
        all_labels = np.concatenate([self.labels, test_labels])
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for metric in metrics:
            accuracies = []
            
            for train_idx, val_idx in kf.split(all_data):
                fold_train_data = all_data[train_idx]
                fold_train_labels = all_labels[train_idx]
                fold_val_data = all_data[val_idx]
                fold_val_labels = all_labels[val_idx]
                
                # Create temporary KNN with this fold's training data
                temp_knn = KNN(fold_train_data.reshape(fold_train_data.shape[0], int(np.sqrt(fold_train_data.shape[1])), int(np.sqrt(fold_train_data.shape[1]))), fold_train_labels)
                
                # Predict and calculate accuracy
                predictions = temp_knn.predict(fold_val_data.reshape(fold_val_data.shape[0], int(np.sqrt(fold_val_data.shape[1])),int(np.sqrt(fold_val_data.shape[1]))), k, metric)
                accuracy = np.mean(predictions == fold_val_labels)
                accuracies.append(accuracy)
            
            # Average accuracy for this metric
            avg_accuracy = np.mean(accuracies)
            print(f"Cross-validation accuracy for {metric}: {avg_accuracy:.4f}")
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_metric = metric
        
        print(f"Best metric: {best_metric} (accuracy: {best_accuracy:.4f})")
        # Set the best metric for this model
        self.metric = best_metric
        return best_metric, best_accuracy
