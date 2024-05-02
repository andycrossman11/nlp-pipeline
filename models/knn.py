from models.nlp_model import NLPModel, majority_vote
import numpy as np


def cluster_assignment(feature_vector: np.ndarray, centroids: dict) -> int:
    """use euclidean distance as measure of similiarity for cluster assignment"""
    assignment = 0
    distance = np.inf
    for label, center in centroids.items():
        # euclidean distance using numpy
        cluster_dist = np.sqrt(np.sum((feature_vector - center)**2))
        if cluster_dist < distance:
            distance = cluster_dist
            assignment = label

    return assignment


class KNN(NLPModel):
    def __init__(self, k: int):
        self.k: int = k
        self.data: np.ndarray = None
        self.labels: np.ndarray

    def train(self, data: np.ndarray, labels: np.ndarray, val_data: np.ndarray = None, val_labels: np.ndarray = None):
        """knn is a lazy learner and just memorizes the dataset. No training"""

        # if k is larger then the dataset, update k to be # of feature vectors in dataset
        if self.k > data.shape[0]:
            self.k = data.shape[0]
        self.data = data
        self.labels = labels

    def eval(self, feature_vector: np.ndarray) -> int:
        """iterate over data and determine which k have the closest euclidean distance. Use majority voting of these k vector's labels to 
        determine prediction label"""
        # initialize distance to be a vector of "n" infinities where n is the # of training instances
        distances = np.full(self.data.shape[0], np.inf)

        # iterate over training set and compute distance from new feature_vector using euclidean distance
        for index, training_sentence in enumerate(self.data):
            distances[index] = np.sqrt(
                np.sum((feature_vector - training_sentence)**2))

        k_smallest_indices = np.argsort(distances)[:self.k]
        labels_of_k_smallest = self.labels[k_smallest_indices]
        return majority_vote(labels_of_k_smallest)
