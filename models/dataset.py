import numpy as np
import random
import nltk

# punkt needed to tokenize
# error with ssl cert. Run these 3 lines below to allow download
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.download('punkt')


def split(data: list[tuple[str, int]], train_tot: float = .7, validation_tot: float = .15, test_tot: float = .15, seed: int = 11) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    # default split of .7 for train, .15 for validation, and .15 for test
    if round(train_tot + validation_tot + test_tot, 3) != 1.000:
        raise Exception("Sum of 3 splits must equal 1.0!")

    if seed != None:
        np.random.seed(seed)
    # shuffle data before split(shuffled along first dimension which is sentences axis)
    np.random.shuffle(data)

    length_of_dataset = len(data)

    # round off the split indices. No index for test needed since it goes to end of matrix
    num_of_train_sents = round(length_of_dataset * train_tot)
    num_of_val_sents = round(length_of_dataset * validation_tot)

    train_dataset = data[0:num_of_train_sents]
    validation_dataset = data[num_of_train_sents:
                              num_of_train_sents + num_of_val_sents]
    test_dataset = data[num_of_val_sents +
                        num_of_train_sents:length_of_dataset]

    train_sents = np.array([data[0] for data in train_dataset])
    train_scores = np.array([data[1] for data in train_dataset], dtype=int)
    val_sents = np.array([data[0] for data in validation_dataset])
    val_scores = np.array([data[1] for data in validation_dataset], dtype=int)
    test_sents = np.array([data[0] for data in test_dataset])
    test_scores = np.array([data[1] for data in test_dataset], dtype=int)
    return ((train_sents, train_scores), (val_sents, val_scores), (test_sents, test_scores))


# return a word to index vocab and a index to word vocab
# the word to index vocab needed to create the feature vectors
# the index to word vocab simply used for testing at end of script
def create_vocabs(tokenized_sentences: list[str]) -> tuple[dict[str:int], dict[int:str]]:
    word_to_index: dict[str:int] = {}
    index_to_word: dict[int:str] = {}
    len_of_vocab: int = 0

    # iterate over tokenized sentences and if a new token is encounted, add it to vocabs with its index value
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_to_index:
                word_to_index[token] = len_of_vocab
                index_to_word[len_of_vocab] = token
                len_of_vocab += 1

    return word_to_index, index_to_word, len_of_vocab


# create the dataset as a matrix using the tokenized sentences and a vocab mapping words to indices
def create_feat_matrix(tokenized_sentences: list[str], vocab: dict[str:int], vocab_len: int) -> np.ndarray:

    # get dimensions of the feature matrix
    num_sentences: int = len(tokenized_sentences)

    # now initialie the feature matrix using num_sentences by len_of_vocab for a 2d array
    feature_matrix = np.zeros((num_sentences, vocab_len))
    # iterate over sentences and corresponding tokens again to fill in matrix
    for sent_num, sentence in enumerate(tokenized_sentences):
        for token in sentence:
            # use vocab to increment the right index in each row for each token in a given sentence
            # notice that the second index value uses vocab[token]!!
            if token in vocab:
                feature_matrix[sent_num][vocab[token]] += 1

    return feature_matrix


def tokenize(sentences: list[str]) -> list[list[str]]:
    tokenized_sentences: list[list[str]] = []
    stem_obj = nltk.stem.PorterStemmer()

    # iterate over sentences to get tokens
    for sentence in sentences:
        # nltk.word_tokenize returns a list of the tokens from a string
        word_tokens = nltk.word_tokenize(sentence)

        # stem and lowercase the tokens and add to the list
        tokenized_sentences.append(
            [stem_obj.stem(token).lower() for token in word_tokens if token.isalpha()])

    return tokenized_sentences


def fit_tf_idf(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idf_vect = np.zeros(shape=(1, matrix.shape[1]))

    for sent_feat_vect in matrix:
        # if token occurs in sent_feat_vect, increment idf_vect. otherwise, retain previous idf value
        idf_vect = np.where(sent_feat_vect >= 1, idf_vect + 1, idf_vect)

    # now update idf in place. NOTE THAT BY ABOVE PRINT STATEMENT, 0 will be selected if word not present
    idf_vect = np.where(
        idf_vect >= 1, np.log10(matrix.shape[0] / idf_vect), 0)

    # feature matrix is already the if values! Just multiply each feature vector
    # feature_matrix shape = 3000 x 4273. idf_vect shape = 1 x 4273.
    # numpy broadcasting makes second arg 3000 x 4273 by replicating the original idf_vect
    matrix = matrix * idf_vect
    return matrix, idf_vect


def fit_prune(matrix: np.ndarray, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    # now find top k features and only keep those values
    feature_counts = np.sum(matrix > 0, axis=0)
    if top_k < matrix.shape[1]:
        top_columns_indices = np.sort(np.argsort(feature_counts)[-top_k:])
    else:
        print(
            f"NO PRUNING SINCE PROVIDED {top_k}, WHICH IS GREATER THAN NUMBER OF FEATURES, {matrix.shape[1]}")

    matrix = matrix[:, top_columns_indices]
    return matrix, top_columns_indices


class Dataset:
    def __init__(self, sentences: list[str], scores: list[int]):
        tokenized_sentences: list[list[str]] = tokenize(sentences)

        # now we run alg with dynamic programming to get our dataset's vocabulary(word to index in feature vector
        # and get inverse dictionary for testing
        self.word_to_index, index_to_word, self.len_of_vocab = create_vocabs(
            tokenized_sentences)

        # then use this vocab to create num_of_sentences by vocab_size sized matrix
        self.feature_matrix: np.ndarray = create_feat_matrix(
            tokenized_sentences, self.word_to_index, self.len_of_vocab)

        # store scores for future assignments
        self.scores = np.array(scores)
        self.length_of_dataset = len(scores)

        # generate 5 random numbers(indices) to extract their feature vecotrs after matrix created
        random_numbers = [random.randrange(
            0, len(sentences) - 1) for _ in range(5)]

        # print sentence and corresponding feat vector
        # for index in random_numbers:
        #     print(
        #         f"Sentence: {sentences[index]}\nFeature Vector: {self.feature_matrix[index]}")
        #     sent_token_indices = np.where(
        #         self.feature_matrix[index] == 1)[0].tolist()
        #     print(f"NON-ZERO INDICES: {sent_token_indices}")
        #     reconstructed_sentence = [index_to_word[index]
        #                               for index in sent_token_indices]
        #     print(f"TOKENS: {reconstructed_sentence}\n\n\n")

    def get_vocab(self) -> dict[str:int]:
        return self.word_to_index
