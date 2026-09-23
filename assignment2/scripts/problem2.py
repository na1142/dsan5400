from nb import build_dataframe, train_nb, test, sklearn_nb

training_df, test_df = build_dataframe("data")
vocabulary, priors, likelihoods = train_nb(training_df)

custom_preds = test(test_df, vocabulary, priors, likelihoods)
sklearn_preds = sklearn_nb(training_df, test_df)

agreement = sum(1 for c, s in zip(custom_preds, sklearn_preds) if c == s)
print(f"Predictions match on {agreement} out of {len(custom_preds)} documents")