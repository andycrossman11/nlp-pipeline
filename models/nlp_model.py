from abc import ABC, abstractmethod
import numpy as np


def accuracy(predictions: np.array, true_labels: np.array) -> float:
    num_correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = round(100 * (num_correct / total), 2)
    return accuracy


def majority_vote(target: np.ndarray) -> int:
    counts = np.bincount(target)
    return int(np.argmax(counts))


class NLPModel(ABC):
    # interface to nlp model to make function calling easier for pipeline class
    def __init__(self):
        pass

    @abstractmethod
    def train(self, feature_matrix: np.ndarray, target: np.ndarray, val_data: np.ndarray = None, val_labels: np.ndarray = None):
        pass

    @abstractmethod
    def eval(self, feature_vector: np.ndarray) -> int:
        pass
