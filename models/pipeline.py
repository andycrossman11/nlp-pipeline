from models.nlp_model import NLPModel
from models.dataset import Dataset, tokenize, create_feat_matrix, fit_tf_idf, fit_prune
import numpy as np
import time

model_types = ["Decision Tree"]


class Pipeline():
    """pipeline to manage ingesting sentences, processing these sentences into feature vectors and saving the necessary objects
    to transform evaluation data in the same format, training model, and evaluating with that model"""

    def __init__(self, model: NLPModel, data: np.ndarray, labels: np.ndarray, tf_idf: bool = False, top_k: int = -1, validation_set: tuple = None):
        if not (isinstance(model, NLPModel)):
            raise Exception(
                f"model type {model} not in available types: {model_types}")

        # dataset handles tokenizing text and transforming data and labels to np.ndarray
        dataset = Dataset(data, labels)
        self.vocab_len = dataset.len_of_vocab
        self.vocab = dataset.get_vocab()

        # start time to train
        start = time.time()

        self.tf_idf: bool = tf_idf

        # if tf_idt and top-k selection is requested than transform the feature vecotrs accordingly
        if tf_idf:
            data, self.idf_vect = fit_tf_idf(dataset.feature_matrix)
        else:
            data = dataset.feature_matrix

        if top_k > -1 and top_k < data.shape[1]:
            self.top_k_set: bool = True
            data, self.top_indices = fit_prune(data, top_k)
        else:
            self.top_k_set: bool = False

        self.model = model

        # if validation during training is requested than pass the set to the model's training function
        if validation_set != None:
            tokenized_val_sents = tokenize(validation_set[0])
            val_matrix = create_feat_matrix(
                tokenized_val_sents, self.vocab, self.vocab_len)
            # tf-idf transform the test data if pipeline utilizes tf-idt
            if self.tf_idf:
                val_matrix = val_matrix * self.idf_vect

            # prune to the top features based on training data pruning
            if self.top_k_set:
                val_matrix = val_matrix[:, self.top_indices]
            self.model.train(data, labels, val_data=val_matrix,
                             val_labels=validation_set[1])
        else:
            self.model.train(data, labels)

        # end time of training
        end = time.time()

        time_to_train = end - start

        print(f"\tTime to Train: {time_to_train} seconds")

        self.trained: bool = True

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        if not (self.trained):
            raise Exception("Error! Please train pipeline with init first!")
        # tokenize sentences
        tokenized_sentences = tokenize(test_data)

        # create matrix of feature vectors of tf values from training data vocab
        test_matrix = create_feat_matrix(
            tokenized_sentences, self.vocab, self.vocab_len)

        # tf-idf transform the test data if pipeline utilizes tf-idt
        if self.tf_idf:
            test_matrix = test_matrix * self.idf_vect

        # prune to the top features based on training data pruning
        if self.top_k_set:
            test_matrix = test_matrix[:, self.top_indices]

        # iterate over transformed feature vecotrs of test data and infer on each to get a label prediction
        predictions = np.zeros((test_data.shape[0]))

        # start time to predict
        start = time.time()
        for index, feature_vector in enumerate(test_matrix):
            predictions[index] = self.model.eval(feature_vector)

        # end time for prediction. Divide prediction time by size of test_data to get average time to classify new feature vector
        end = time.time()
        time_to_predict = (end - start) / test_data.shape[0]
        print(f"\t\tTime to Predict: {time_to_predict} seconds")
        return predictions
