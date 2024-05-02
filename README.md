## Project Description:

- First reads .txt files with sentences and a corresponding label, 0 for negative sentiment and 1 for positive sentiment. (Example files can be found [here](http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences))
- Tokenizes the sentences with NLTK and creates a vocabulary.
- Sentences are transformed into 1-gram feature vectors. Then, I transform the vectors with TF-IDF and select the top-k tokens from this transformed dataset.
- Custom implementations for KNN, Decision Tree, and a small Neural Net.
- I train each model with the original 1-gram dataset and the transformed data.
- I then report time to train, time to predict, and my implementation's accuracy against scikit-learn (when applicable).
- My models package allows for easy, modular machine learning pipelines

## File Descriptions:

- `driver.py`: The main script to execute and display experiment results.
- `dataset.py`: Text processing functions for splitting dataset, tokenizing, vocabulary creation, TF-IDF, and top-k feature selection.
- `nlp_model.py`: Interface for my implementations to make execution easier in other scripts. Also has shared functions for accuracy computation and majority voting for classification.
- `fnn.py`: My implementation of a fully connected neural net.
- `knn.py`: My implementation of a k nearest neighbor classifier.
- `decision_tree.py`: My implementation of a decision tree.
- `pipeline.py`: A wrapper around the whole processing, training, and evaluation processes. Takes in a model and training data to process the data accordingly (with flags set like top_k and tf_idf), then train model. Predict takes in test data and processes it according to train data processing and classifies based on trained model.

## Libraries Needed (all in `requirements.txt`):

- scikit-learn (used for validation of my model implementation)
- PyTorch
- NumPy
- NLTK

**Expects a directory called "sentiment_labelled_sentences" of .txt files of sentence label pairs at root**  
**To execute, just run `python driver.py`**
