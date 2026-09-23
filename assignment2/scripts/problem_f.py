from nb import build_dataframe, train_nb, test

training_df, test_df = build_dataframe("data")
vocabulary, priors, likelihoods = train_nb(training_df)

predictions = test(test_df, vocabulary, priors, likelihoods)

id_to_author = {0: "kennedy", 1: "johnson"}
for i in range(len(test_df)):
    predicted_label = predictions[i]
    predicted_author = id_to_author[predicted_label]
    print(f"Doc no. {i} predicted author = {predicted_author}")