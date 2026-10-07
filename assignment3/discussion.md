## Problem 1

#### Part A

The build_dataframe function is built very similarly, except I explicitly defined a variable storing the author name for the case where the directory is not kennedy or johnson (the unlabeled folder), since there the author has to be pulled from the file name instead of the folder name. While professor's code performs that task inside panda's data frame function, I explicity defined it in advance because it helps me with readability as someone who does not have a ton of experience writing elaborate Python code.

In the train_nb function, professor built the vocabulary dictionary in fewer lines than I did, I had two do it in two steps because I had to conceptually break it into two steps to figure out how to build it. I think I like the way I built the priors list by just using the normalize argument of value counts instead of explicitly dividing counts by the total number of documents in the training set, I believe it is cleaner. For computing word counts per class, I had to create two separate filtered data frames for each class and iterate over them separately to fill the word counts per class dictionary. Professor wrote his entire logic to perform this task in one line, which is more compact but his code is very hard for me understand, albeit more "pythonic." 

In general, professor tends to write shorter code in fewer lines whereas I have to lay out everything one by one for my own understanding. A trivial example in the testing function can be seen where professor directly loops over text.split(), whereas I first created a variable that stores the split text and then looped over it. I prefer my way because I can reuse the variable if needed.

