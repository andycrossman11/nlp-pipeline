import os
from models.decision_tree import DecisionTree
from models.knn import KNN
from models.fnn import FNN
from models import accuracy
from models.dataset import split
from models import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# given a folder path, read all the .txt files, where each file has a sentence and score(0 or 1) on each line
def read_folder(folder_path: str) -> list[tuple[str, int]]:
    total_data: list[tuple[str, int]] = []
    all_txt = [f"{folder_path}/{file}" for file in os.listdir(
        folder_path) if file.endswith(".txt")]

    for file_path in all_txt:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.replace('\n', '')
                # From readme of dataset, we know last token is the score.
                # Use this info to separate the sentences and scores and store in their corresponding lists
                data: tuple = (line[0:len(line) - 2].rstrip(),
                               int(line[len(line) - 1]))
                total_data.append(data)

    return total_data


def experiment_results(pipeline: Pipeline, train_data: tuple[np.ndarray, np.ndarray], val_data: tuple[np.ndarray, np.ndarray], test_data: tuple[np.ndarray, np.ndarray], sklearn_model=None) -> None:
    # run experiment for a data processing type and model type combo

    print("\tTrain Data:")
    train_acc = accuracy(pipeline.predict(train_data[0]), train_data[1])
    print(f"\t\tMy implementation Accuracy: {train_acc}%")
    if sklearn_model != None:
        sklearn_score(sklearn_model, train_data, train_data)

    print("\tValidation Data:")
    val_acc = accuracy(pipeline.predict(val_data[0]), val_data[1])
    print(f"\t\tMy implementation Accuracy: {val_acc}%")
    if sklearn_model != None:
        sklearn_score(sklearn_model, train_data, val_data)

    print("\tTest Data:")
    test_acc = accuracy(pipeline.predict(test_data[0]), test_data[1])
    print(f"\t\tMy implementation Accuracy: {test_acc}%")
    if sklearn_model != None:
        sklearn_score(sklearn_model, train_data, test_data)


def sklearn_score(model, train_data: tuple[np.ndarray, np.ndarray], test_data: tuple[np.ndarray, np.ndarray], top_k: int = -1) -> None:
    # given a sklearn model and some sentence data, first tf-idf and possibly top-k to create feature vectors.
    # Train and test the sklearn model on these feature vectors

    train_sents, train_scores = train_data[0], train_data[1]
    test_sents, test_scores = test_data[0], test_data[1]

    if top_k > -1:
        # use sklearn's tf-idf implementation
        vectorizer = TfidfVectorizer()
        train_sents = vectorizer.fit_transform(train_sents)

        # Select top_k features after tf_idf transform with chi2
        selector = SelectKBest(chi2, k=top_k)
        train_sents = selector.fit_transform(train_sents, train_scores)

        # transform test sentences using tf-idf
        test_sents = selector.transform(vectorizer.transform(test_sents))
    else:
        # Initialize CountVectorizer as it is same algorithm as HW2 vector creation
        vectorizer = CountVectorizer()
        train_sents = vectorizer.fit_transform(train_sents)

        test_sents = vectorizer.transform(test_sents)

    # fit model, get predictions from trained model, compute accuracy
    model.fit(train_sents, train_scores)

    predictions = model.predict(test_sents)

    accuracy = round(100 * accuracy_score(test_scores, predictions), 2)
    print(f"\t\tSklearn Accuracy: {accuracy}%")


if __name__ == "__main__":
    # where TOP_K is the top k tokens from tf-idf to be kept in feature vector and NUM_NEIGHBORS is neighbors in knn algorithm
    TOP_K = 50
    NUM_NEIGHBORS = 10
    total_data = read_folder("./sentiment_labelled_sentences")

    # get train,validation,test split from function implemented in dataset.py
    train_data, val_data, test_data = split(
        total_data, train_tot=.7, validation_tot=.15, test_tot=.15)

    # declare my nlp implementations
    knn = KNN(k=NUM_NEIGHBORS)
    dtree = DecisionTree()

    # declare sklearn models
    skl_dtree = DecisionTreeClassifier()
    skl_knn = KNeighborsClassifier(n_neighbors=NUM_NEIGHBORS)

    print("KNN w/ Original HW2 Vectors:")
    knn_pipeline = Pipeline(
        knn, train_data[0], train_data[1]
    )
    experiment_results(knn_pipeline, train_data, val_data,
                       test_data, sklearn_model=skl_knn)

    print("KKN w/ TF-IDF and Top-K Selected Feature Vectors:")
    knn_pipeline = Pipeline(
        knn, train_data[0], train_data[1], top_k=TOP_K, tf_idf=True
    )
    experiment_results(knn_pipeline, train_data, val_data,
                       test_data, sklearn_model=skl_knn)

    print("Decision Tree w/ Original HW2 Vectors:")
    dtree_pipeline = Pipeline(
        dtree, train_data[0], train_data[1])
    experiment_results(dtree_pipeline, train_data, val_data,
                       test_data, sklearn_model=skl_dtree)

    print("Decision Tree w/ TF-IDF and Top-K Selected Feature Vectors:")
    dtree_pipeline = Pipeline(
        dtree, train_data[0], train_data[1], tf_idf=True, top_k=TOP_K)
    experiment_results(dtree_pipeline, train_data, val_data,
                       test_data, sklearn_model=skl_dtree)

    # BONUS -> display validation accuracy and implemented early stopping for fully connected neural net.
    # Displays final accuracy on test set as well
    print("Neural Net w/ Original HW2 Vectors:")
    fnn = FNN(75)
    fnn_pipeline = Pipeline(
        fnn, train_data[0], train_data[1], validation_set=val_data)
    experiment_results(fnn_pipeline, train_data, val_data, test_data)

    print("Neural Net w/ TF-IDF and Top-K Selected Feature Vectors:")
    fnn = FNN(75)
    fnn_pipeline = Pipeline(
        fnn, train_data[0], train_data[1], tf_idf=True, top_k=TOP_K, validation_set=val_data)
    experiment_results(fnn_pipeline, train_data, val_data, test_data)
