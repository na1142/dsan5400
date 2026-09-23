from nb import build_dataframe, train_nb, test, sklearn_nb, get_metrics
import matplotlib.pyplot as plt
import seaborn as sns

training_df, test_df = build_dataframe("data")
vocabulary, priors, likelihoods = train_nb(training_df)

custom_preds = test(test_df, vocabulary, priors, likelihoods)
acc, f1, custom_conf = get_metrics(test_df["author"], custom_preds)
print("Custom NB accuracy:", acc, "F1:", f1)

sklearn_preds = sklearn_nb(training_df, test_df)
sklearn_acc, sklearn_f1, sklearn_conf = get_metrics(test_df["author"], sklearn_preds)
print("sklearn NB accuracy:", sklearn_acc, "F1:", sklearn_f1)


def save_confusion_matrix(conf_matrix_data, labels, title, save_path):
    plt.figure()
    plt.title(title)
    axis = sns.heatmap(conf_matrix_data, annot=True, fmt="d", cmap="Blues")
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.savefig(save_path)
    plt.show()


save_confusion_matrix(custom_conf, [0, 1], "Custom NB Confusion Matrix", "conf_custom.jpg")
save_confusion_matrix(sklearn_conf, [0, 1], "Sklearn NB Confusion Matrix", "conf_sklearn.jpg")