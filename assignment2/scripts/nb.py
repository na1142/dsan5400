import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=["author"])
    df_test = pd.DataFrame(columns=["author"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f, encoding="utf-8") as fp:
                text = fp.read()
                if dir_name in ("kennedy", "johnson"):
                    one_row = pd.DataFrame({"author": [dir_name], "text": [text]})
                    df = pd.concat([df, one_row], ignore_index=True)
                else:
                    author_name = f.stem.split("_")[-1]
                    one_row = pd.DataFrame({"author": [author_name], "text": [text]})
                    df = pd.concat([df, one_row], ignore_index=True)
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))
    return df_train, df_test


def train_nb(df, alpha=0.1):
    """
    Takes as input a pandas DataFrame containing Federalist
    files text to determine priors and likelihoods
    :param df: a pandas DataFrame
    :return: two numpy arrays for the priors and likelihoods
    """
    all_words = [token for text in df['text'] for token in text.split()]
    vocabulary = {word: index for index, word in enumerate(set(all_words))}
    n_docs = df.shape[0]
    n_classes = df["author"].nunique()

    priors = df['author'].value_counts(normalize=True).sort_index().tolist()

    training_matrix = np.zeros((n_docs, len(vocabulary)))
    for row_index, text in enumerate(df['text']):
        for word in text.split():
            col_index = vocabulary[word]
            training_matrix[row_index, col_index] = text.split().count(word)

    word_counts_per_class = {0: {}, 1: {}}

    df_0 = df[df['author'] == 0]
    df_1 = df[df['author'] == 1]

    for text in df_0['text']:
        token = text.split()
        for word in token:
            word_counts_per_class[0][word] = word_counts_per_class[0].get(word, 0) + token.count(word)

    for text in df_1['text']:
        token = text.split()
        for word in token:
            word_counts_per_class[1][word] = word_counts_per_class[1].get(word, 0) + token.count(word)

    likelihoods = np.zeros((n_classes, len(vocabulary)))

    for C in {0, 1}:
        total_word_count_for_class = sum(word_counts_per_class[C].values())
        denominator = total_word_count_for_class + alpha * len(vocabulary)

        for word in vocabulary:
            col_index = vocabulary[word]
            count = word_counts_per_class[C].get(word, 0)
            numerator = count + alpha
            likelihoods[C, col_index] = numerator / denominator

    return vocabulary, priors, likelihoods


def test(df, vocabulary, priors, likelihoods):
    """
    Takes as input a pandas DataFrame representing the disputed Federalist
    Papers and returns predictions for every text document
    :param df: a pandas DataFrame
    :return: a numpy array of predictions
    """
    n_classes = len(priors)
    class_predictions = []
    for text in df["text"]:
        test_vector = np.zeros(shape=(len(vocabulary)))
        tokens = text.split()
        for word in tokens:
            if word in vocabulary:
                col_index = vocabulary[word]
                test_vector[col_index] = tokens.count(word)

        preds = []
        for C in range(n_classes):
            score = np.log(priors[C]) + np.sum(test_vector * np.log(likelihoods[C]))
            preds.append(score)

        yhat = np.argmax(preds)
        class_predictions.append(yhat)
    return np.array(class_predictions)


def sklearn_nb(training_df, test_df):
    """
    Performs Naive Bayes classification using scikit-learn implementation
    :param training_df: training data
    :param test_df: test data
    :return: predictions
    """
    vectorizer = CountVectorizer()
    vectorizer.fit(training_df["text"])

    training_data = vectorizer.transform(training_df["text"])
    training_data.toarray()

    test_data = vectorizer.transform(test_df["text"])
    test_data.toarray()

    nb_classifier = MultinomialNB()
    nb_classifier.fit(training_data, training_df["author"])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb


def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    accuracy = metrics.accuracy_score(true, preds)
    f1_score = metrics.f1_score(true, preds)
    conf_matrix = metrics.confusion_matrix(true, preds)

    return accuracy, f1_score, conf_matrix


def plot_confusion_matrix(conf_matrix_data, labels):
    """
    Takes as input confusion matrix data from get_metrics() and prints out a
    confusion matrix
    :param conf_matrix_data:
    :return: None
    """
    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data, annot=True, fmt="d", cmap="Blues")
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()

    training_df, test_df = build_dataframe(args.indir)
    vocabulary, priors, likelihoods = train_nb(training_df)
    class_predictions = test(test_df, vocabulary, priors, likelihoods)
    acc, f1, conf = get_metrics(test_df["author"], class_predictions)
    print("Custom NB accuracy:", acc, "F1:", f1)
    plot_confusion_matrix(conf, [0, 1])

    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_acc, sklearn_f1, sklearn_conf = get_metrics(test_df["author"], sklearn_preds)
    print("sklearn NB accuracy:", sklearn_acc, "F1:", sklearn_f1)
