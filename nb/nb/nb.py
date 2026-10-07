import pandas as pd
import numpy as np

class NaiveBayes:

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def train(self, df):
        """
        Takes as input a pandas DataFrame containing Federalist
        files text to determine priors and likelihoods
        :param df: a pandas DataFrame
        :return: two numpy arrays for the priors and likelihoods
        """
        self.vocabulary = {word: idx for idx, word in
            enumerate(set(" ".join(df['text'].tolist()).split()))}
        n_docs = df.shape[0]
        n_classes = df.author.nunique()
        self.priors = np.array([sum(df['author'] == auth) / n_docs for auth in range(
            n_classes)])
        # Create a matrix containing all 0s called training_matrix of size (n_docs,
        # len(vocabulary)), then fill it with the counts of each word for each
        # document
        # this is the bag-of-words matrix for all the documents
        training_matrix = np.zeros(shape=(n_docs, len(self.vocabulary)))
        for idx, document in enumerate(df['text'].tolist()):
            for token in document.split():
                j = self.vocabulary[token]
                training_matrix[idx, j] += 1
        # get word counts for both classes
        word_counts_per_class = {auth: np.sum(training_matrix[np.where(
            df['author'] == auth)]) for auth in range(n_classes)}
        self.likelihoods = np.zeros(shape=(n_classes, len(self.vocabulary)+1))
        
        for token, idx in self.vocabulary.items():
            for auth in range(n_classes):
                count_token_idx_in_class_auth = sum(np.squeeze(training_matrix[
                        np.where(df['author'] == auth), idx]))
                self.likelihoods[auth, idx] = (self.alpha +
                    count_token_idx_in_class_auth) / \
                    (self.alpha * (len(self.vocabulary) + 1) + word_counts_per_class[auth])
        for auth in range(n_classes):
            self.likelihoods[auth, -1] = self.alpha / (
            self.alpha * (len(self.vocabulary) + 1) + word_counts_per_class[auth])
        return self.vocabulary, self.priors, self.likelihoods


    def test(self, df):
        """
        Takes as input a pandas DataFrame representing the disputed Federalist
        Papers and returns predictions for every text document
        :param df: a pandas DataFrame
        :return: a numpy array of predictions
        """
        self.class_predictions = []
        for text in df['text']:
            test_vector = np.zeros(shape=(len(self.vocabulary)+1))
            for token in text.split():
                # skip the words that do not appear in the training corpus
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    test_vector[idx] += 1
                else:
                    test_vector[-1] += 1
            # compute predictions p(y|test)
            preds = test_vector.dot(np.log(self.likelihoods).T) + np.log(self.priors)
            yhat = np.argmax(preds)
            self.class_predictions.append(yhat)
        return self.class_predictions