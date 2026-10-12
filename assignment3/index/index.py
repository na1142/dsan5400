import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from collections import Counter


def load_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for doc in root.iter("doc"):
        row = {}
        for child in doc:
            tag_name = child.tag
            text = child.text
            if text is None:
                text = ""
            text = text.strip()
            row[tag_name] = text
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


df = load_xml("20newsgroups-initial.xml")

class Tokenizer:
    def tokenize(self, text):
        raise NotImplementedError
    def tokenize_dataframe(self, data_frame):
        token_dict = {}
        for index, row in data_frame.iterrows():
            raw_tokens = self.tokenize(row['msgtext'])
            tokens = []
            for raw_token in raw_tokens:
                lowered = raw_token.lower()
                token = lowered.strip(string.punctuation)
                if token != "":
                    tokens.append(token)
            token_dict[row['docid']] = tokens
        return token_dict

class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text):
        tokens = text.split()
        return tokens

class NLTKTokenizer(Tokenizer):
    def tokenize(self, text):
        tokens = word_tokenize(text)
        return tokens

class NgramTokenizer(Tokenizer):
    def __init__(self, n=4):
        self.n = n
    def tokenize(self, text):
        tokens = []
        last_start = len(text) - self.n + 1
        for i in range(last_start):
            gram = text[i:i + self.n]
            tokens.append(gram)
        return tokens

def stem_tokens(token_dict):
    stemmer = PorterStemmer()
    stemmed_dict = {}
    for docid in token_dict:
        tokens = token_dict[docid]
        stemmed_tokens = []
        for token in tokens:
            stem = stemmer.stem(token)
            stemmed_tokens.append(stem)
        stemmed_dict[docid] = stemmed_tokens
    return stemmed_dict

def invert_index(stemmed_dict):

    index = {}

    for docid, text in stemmed_dict.items():
        term_frequency = Counter(text)

        for stem, count in term_frequency.items():
            if stem not in index:
                index[stem] = [(docid, count)]
            else:
                index[stem].append((docid, count))
    return index

def search(query, tokenizer, index):
    stemmer = PorterStemmer()
    raw_tokens = tokenizer.tokenize(query)
    result = None
    for raw_token in raw_tokens:
        lowered = raw_token.lower()
        token = lowered.strip(string.punctuation)
        stem = stemmer.stem(token)
        postings = index.get(stem, [])
        docids = set()
        for docid, count in postings:
            docids.add(docid)
        if result is None:
            result = docids
        else:
            result = result.intersection(docids)
    if result is None:
        result = set()
    return sorted(result, key=int)

whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize_dataframe(df)
whitespace_stemmed = stem_tokens(whitespace_tokens)
whitespace_index = invert_index(whitespace_stemmed)
print(search('system', whitespace_tokenizer, whitespace_index))

nltk_tokenizer = NLTKTokenizer()
nltk_tokens = nltk_tokenizer.tokenize_dataframe(df)
nltk_stemmed = stem_tokens(nltk_tokens)
nltk_index = invert_index(nltk_stemmed)
print(search('system', nltk_tokenizer, nltk_index))

ngram_tokenizer = NgramTokenizer(n=4)
ngram_tokens = ngram_tokenizer.tokenize_dataframe(df)
ngram_stemmed = stem_tokens(ngram_tokens)
ngram_index = invert_index(ngram_stemmed)
print(search('system', ngram_tokenizer, ngram_index))

print(search('compatibility', whitespace_tokenizer, whitespace_index))
print(search('compatibility', nltk_tokenizer, nltk_index))
print(search('compatibility', ngram_tokenizer, ngram_index))

system_whitespace = set(search('system', whitespace_tokenizer, whitespace_index))
compat_whitespace = set(search('compatibility', whitespace_tokenizer, whitespace_index))
print(system_whitespace.intersection(compat_whitespace))

system_nltk = set(search('system', nltk_tokenizer, nltk_index))
compat_nltk = set(search('compatibility', nltk_tokenizer, nltk_index))
print(system_nltk.intersection(compat_nltk))

system_ngram = set(search('system', ngram_tokenizer, ngram_index))
compat_ngram = set(search('compatibility', ngram_tokenizer, ngram_index))
print(system_ngram.intersection(compat_ngram))