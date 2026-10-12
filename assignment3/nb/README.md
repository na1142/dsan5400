# Naive Bayes Classifier

This project is a Naive Bayes classifier that reads presidential speeches and guesses whether each one was written by Kennedy or Johnson. It uses Lidstone smoothing so words the model has never seen still get a small probability instead of zero. I set it up as a Python package so the code is easy to import and test.

## How it is organized

The NaiveBayes class lives in nb/nb.py and has two methods. The train method learns the priors and likelihoods from labeled speeches. The test method takes new speeches and predicts an author for each one. The likelihood matrix has one extra column that holds the probability for any word the model did not see during training.

The code that loads the speech files into a pandas DataFrame is in nb/utils/load_data.py. The command line script that ties everything together is in nb/bin/main.py. The unit tests are in the tests folder.

## Setup

First create the conda environment from the environment.yml file and activate it. Then go to the folder that has setup.py in it and install the package:

    pip install -e ".[dev]"

## How to run it

Run the main script and point it at your data folder:

    python nb\bin\main.py -f data

The data folder needs three subfolders called kennedy and johnson and unlabeled. The kennedy and johnson folders hold the training speeches. The unlabeled folder holds the speeches the model predicts on. The data folder is ignored by git, so you have to add your own copy.

The script prints the priors and the likelihoods and then the predicted author for each unlabeled speech. A 0 means Kennedy and a 1 means Johnson. You can also set the smoothing value with the optional -a command which defaults to 0.1:

    python nb\bin\main.py -f data -a 1

## How to test it

Run this from the folder that has pytest.ini in it:

    pytest

There are three tests. They check the priors and the likelihoods and that each class's probabilities add up to 1. They use a tiny example of five short movie reviews from Jurafsky and Martin, saved in tests/data/docs.txt so the answers can be worked out by hand.