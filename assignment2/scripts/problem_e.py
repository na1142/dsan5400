from nb import build_dataframe, train_nb, test, get_metrics
from sklearn.model_selection import train_test_split

training_df, test_df = build_dataframe("data")

train_split, eval_split = train_test_split(
    training_df, test_size=0.2, random_state=42, stratify=training_df["author"]
)

vocabulary, priors, likelihoods = train_nb(train_split)
print("Priors:", priors)
print("Likelihoods shape:", likelihoods.shape)

for alpha_value in [0.01, 0.1, 1, 10]:
    vocabulary, priors, likelihoods = train_nb(train_split, alpha=alpha_value)
    predictions = test(eval_split, vocabulary, priors, likelihoods)
    acc, f1, conf = get_metrics(eval_split["author"], predictions)
    print(f"alpha={alpha_value}: accuracy={acc}, f1={f1}")