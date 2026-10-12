## Problem 1

#### Part A

The build_dataframe function is built very similarly, except I explicitly defined a variable storing the author name for the case where the directory is not kennedy or johnson (the unlabeled folder), since there the author has to be pulled from the file name instead of the folder name. While professor's code performs that task inside panda's data frame function, I explicity defined it in advance because it helps me with readability as someone who does not have a ton of experience writing elaborate Python code.

In the train_nb function, professor built the vocabulary dictionary in fewer lines than I did, I had two do it in two steps because I had to conceptually break it into two steps to figure out how to build it. I think I like the way I built the priors list by just using the normalize argument of value counts instead of explicitly dividing counts by the total number of documents in the training set, I believe it is cleaner. For computing word counts per class, I had to create two separate filtered data frames for each class and iterate over them separately to fill the word counts per class dictionary. Professor wrote his entire logic to perform this task in one line, which is more compact but his code is very hard for me understand, albeit more "pythonic." 

In general, professor tends to write shorter code in fewer lines whereas I have to lay out everything one by one for my own understanding. A trivial example in the testing function can be seen where professor directly loops over text.split(), whereas I first created a variable that stores the split text and then looped over it. I prefer my way because I can reuse the variable if needed.

#### Part C

Here is the output after running pytest:

(dsan5400) PS C:\Users\naufa\OneDrive\Documents\Georgetown DSAN\DSAN 5400\assignment3\assignment3\nb> pytest
============================= test session starts ==============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\naufa\OneDrive\Documents\Georgetown DSAN\DSAN 5400\assignment3\assignment3\nb
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.14.2
collected 3 items                                                               

tests\test_naive_bayes.py ...                                             [100%]

============================== 3 passed in 0.55s ===============================
(dsan5400) PS C:\Users\naufa\OneDrive\Documents\Georgetown DSAN\DSAN 5400\assignment3\assignment3\nb> 

## Problem 2

#### Part B

The universal property of a tokenizer is that it takes a string of text and returns a list of string tokens so my Tokenizer interface only requires a tokenize(text) method and each subclass has its own splitting logic.

#### Part E

All three indices show the same 10 documents that the query 'system' appears in. It's cool that the n-gram index also returned the same documents, I would think that it might be possible return documents where the word isn't actually there because one its grams may come from a longer word that is not actually 'system.'

#### Part F

In the query for 'compatibility' all three indexes returned the same two documents (8772 and 14398). So just like with system' the tokenizer choice didn’t change the results, even though each one breaks the word up differently.

#### Part G

All three indexes return documents 8772 and 14398 for 'system' and 'compatibility' together. I searched each word on its own and kept only the documents that showed up in both results. The tokenizers all agree here so the tokenizer choice didn't change which documents came back for these queries.

