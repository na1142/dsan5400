import argparse

from nb.utils.load_data import DataLoader
from nb.nb import NaiveBayes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
                        help="Lidstone smoothing parameter (default 0.1)")
    args = parser.parse_args()

    loader = DataLoader(args.indir)
    df_train, df_test = loader.build_dataframe()

    model = NaiveBayes(alpha=args.alpha)
    model.train(df_train)

    print("Priors:", model.priors)
    print("Likelihoods:", model.likelihoods)

    predictions = model.test(df_test)
    print("The predictions are:", predictions)


