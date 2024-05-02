import numpy as np
from abc import ABC, abstractmethod
from models.nlp_model import NLPModel, majority_vote


class Node(ABC):
    # interface for implementations to make decision tree easier to traverse/predict with
    def __init__(self):
        pass

    @abstractmethod
    def traverse(self, data: np.ndarray) -> int:
        pass


class SplitNode(Node):
    # use index and split to go to correct child node based on feature vector given
    def __init__(self, split: float, index: int, children: list[Node]):
        self.split = split
        self.index = index
        self.children: list[Node] = children

    def traverse(self, data: np.ndarray) -> int:
        if data[self.index] <= self.split and self.children[0] != None:
            return self.children[0].traverse(data)
        elif self.children[1] != None:
            return self.children[1].traverse(data)
        else:
            raise Exception("Error! Split node with no children")


class LeafNode(Node):
    # make prediction
    def __init__(self, prediction: int):
        self.prediction = prediction

    def traverse(self, data: np.ndarray):
        return self.prediction


def gini_score(ds: np.ndarray, target: np.ndarray, split: float) -> int:
    """Compute gini impurity for a feature based on a particular split value.
    where ds is a dataset with each feature vector of length 1."""
    less_than_eq: int = 0
    less_than_eq_pos_freq: int = 0
    greater_than_freq: int = 0

    # where x is a point in dataset
    for index, x in enumerate(ds):
        if x <= split:
            less_than_eq += 1
            less_than_eq_pos_freq += int(target[index])
        else:
            greater_than_freq += int(target[index])

    if less_than_eq > 0:
        less_than_pos_prob = less_than_eq_pos_freq / less_than_eq
    else:
        less_than_pos_prob = 0

    less_than_gini = (1 - ((less_than_pos_prob ** 2) +
                      ((1 - less_than_pos_prob) ** 2)))
    less_than_scaling = (less_than_eq / ds.shape[0])

    if ds.shape[0] - less_than_eq > 0:
        greater_than_prob = greater_than_freq / (ds.shape[0] - less_than_eq)
    else:
        greater_than_prob = 0

    greater_than_gini = (1 - ((greater_than_prob ** 2) +
                         ((1 - greater_than_prob) ** 2)))
    greater_than_scaling = ((ds.shape[0] - less_than_eq) / ds.shape[0])
    return ((less_than_scaling * less_than_gini) + (greater_than_scaling * greater_than_gini))


def gini_round(data: np.ndarray, target: np.ndarray, used_attributes: dict) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over unused attributes(indices) find its lowest gini value by:
        - sorting values across all feature vectors at that index
        - computing gini score for each unique value in this sort
        - storing lowest gini value for this index in gini_impurities and that split in gini_splits"""
    gini_impurities = np.full(shape=(data.shape[1]), fill_value=np.inf)
    gini_splits = np.zeros(shape=(data.shape[1]))

    for index, feature in enumerate(data.T):
        if str(index) not in used_attributes.keys():
            # get unique feature values for each feature index
            sorted_feat_vals = np.unique(np.sort(feature))
            for j in range(0, sorted_feat_vals.shape[0] - 1):
                sorted_feat_vals[j] = (
                    sorted_feat_vals[j] + sorted_feat_vals[j+1]) / 2
            if len(sorted_feat_vals) > 1:
                sorted_feat_vals = sorted_feat_vals[:-1]

            for split in sorted_feat_vals:
                # update gini_splits value for feature if new gini score is lower than current value
                gini_impurity = gini_score(feature, target, split)
                if gini_impurity < gini_impurities[index]:
                    gini_impurities[index] = gini_impurity
                    gini_splits[index] = split

    return (gini_impurities, gini_splits)


def train_d_tree(data: np.ndarray, target: np.ndarray, used_attributes: dict) -> Node | None:
    """recursively divide data until a decision tree stopping condition is met:
        - cannot further divide data
        - data is pure"""
    # if no data is left, no node creation to do
    if data.shape[0] < 1:
        return None
    # if data is pure w/ 1, predict 1
    elif np.all(target == 1):
        return LeafNode(1)
    # if data is pure w/ 0, predict 0
    elif np.all(target == 0):
        return LeafNode(0)
    elif len(list(used_attributes.keys())) == used_attributes["length"] + 1:
        return LeafNode(majority_vote(target))

    gini_impurities, gini_splits = gini_round(data, target, used_attributes)

    # use argmin to get index of smallest gini impurity. Use this index to access the corresponding split value.
    selected_index = np.argmin(gini_impurities)
    used_attributes[str(selected_index)] = 1
    split_value = gini_splits[selected_index]

    # split the dataset
    less_than_eq: list = []
    less_than_targets: list = []
    greater_than: list = []
    greater_than_targets: list = []
    for index, feat_vect in enumerate(data):
        if feat_vect[selected_index] <= split_value:
            less_than_eq.append(feat_vect)
            less_than_targets.append(target[index])
        else:
            greater_than.append(feat_vect)
            greater_than_targets.append(target[index])

    left_split, left_target = np.array(
        less_than_eq), np.array(less_than_targets)
    right_split, right_target = np.array(
        greater_than), np.array(greater_than_targets)

    if left_split.shape[0] == data.shape[0] or right_split.shape[0] == data.shape[0]:
        return LeafNode(majority_vote(target))

    # find split nodes through recursive calls
    left_node = train_d_tree(left_split, left_target, used_attributes)
    right_node = train_d_tree(right_split, right_target, used_attributes)
    return SplitNode(split_value, selected_index, children=[left_node, right_node])


class DecisionTree(NLPModel):

    def __init__(self):
        """implemented using gini index for split heuristic"""
        self.rootNode: Node = None

    def train(self, feature_matrix: np.ndarray, target: np.ndarray, val_data: np.ndarray = None, val_labels: np.ndarray = None):
        used_attributes: dict = {"length": feature_matrix.shape[1]}
        self.rootNode: Node = train_d_tree(
            feature_matrix, target, used_attributes)

    def eval(self, feature_vector: np.ndarray) -> int:
        return self.rootNode.traverse(feature_vector)
