from pathlib import Path
import pandas as pd
from nb.nb import NaiveBayes
import pytest

def __test_dataframe():
    path = Path(__file__).parent / "data" / "docs.txt"
    f = path.read_text()    

    labels = []
    texts = []
    for line in f.splitlines():
        if line.strip() == "":
            continue
        label, text = line.split(",", 1)
        labels.append(0 if label.strip() == "-" else 1)
        texts.append(text.strip())

    df = pd.DataFrame({"author": labels, "text": texts})

    return df

def test_nb_priors():
    df = __test_dataframe()

    model = NaiveBayes(alpha=1)
    model.train(df)
    assert model.priors == pytest.approx((0.6,0.4))

def test_nb_likelihoods():
    df = __test_dataframe()

    model = NaiveBayes(alpha=1)
    model.train(df)

    expected = {
    "just":        (2/35, 1/30),
    "plain":       (2/35, 1/30),
    "boring":      (2/35, 1/30),
    "entirely":    (2/35, 1/30),
    "predictable": (2/35, 1/30),
    "and":         (3/35, 1/30),
    "lacks":       (2/35, 1/30),
    "energy":      (2/35, 1/30),
    "no":          (2/35, 1/30),
    "surprises":   (2/35, 1/30),
    "very":        (2/35, 2/30),
    "few":         (2/35, 1/30),
    "laughs":      (2/35, 1/30),
    "powerful":    (1/35, 2/30),
    "the":         (1/35, 3/30),
    "most":        (1/35, 2/30),
    "fun":         (1/35, 2/30),
    "film":        (1/35, 2/30),
    "of":          (1/35, 2/30),
    "summer":      (1/35, 2/30),
    }

    assert model.likelihoods.shape == (2, 21)

    for word, pair in expected.items():
        col = model.vocabulary[word]
        assert model.likelihoods[:, col] == pytest.approx(pair)

    assert model.likelihoods[:, -1] == pytest.approx((1/35, 1/30))

def test_nb_axis_sum():
    df = __test_dataframe()

    model = NaiveBayes(alpha=1)
    model.train(df)
    assert model.likelihoods[0,:].sum() == pytest.approx(1)
    assert model.likelihoods[1,:].sum() == pytest.approx(1)




