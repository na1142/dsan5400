# Discussion Responses

## Problem 1

### Part B

I would guess Johnson because there are more papers authored by Johnson than Kennedy. In other words, the prior probability for Johnson, $\Pr(\textnormal{Johnson})$ is higher.

### Part C

Kennedy's documents tend to be shorter in length, as the average word count of Kennedy's documents is about 2879 words compared to Johnson's 3620. Both author's most commonly used words are function words, which is to be expected. 

Kennedy's speeches usually open by naming a bunch of specific officials or dignitaries in the room so each one looks a little different depending on the audience. Johnson's openings are way more repetitive and a lot of them literally start with "THE PRESIDENT. Good afternoon, ladies and gentlemen" or "My fellow Americans" almost like a script. The closings tell a similar story, Kennedy tends to end on a theme, something about freedom, peace, or duty and the wording changes each time. Johnson's endings are much more standardized with "Thank you, Mr. President" showing up constantly. Overall it looks like Johnson's texts are closer to transcribed press conferences while Kennedy's are more like standalone crafted speeches.

#### Part E

The prior probabilities are 0.358 for Kennedy and 0.642 for Johnson. The shape of the likelihood matrix is $\texttt{(2, len(vocabulary))}$ which is $\texttt{(2, 21791)}$. I varied alpha from 0.01, 0.1, 1 and 10, and it looked like the accuracy and f1 scores peaked at 1 after dropping back down. If not 1, peak model performance occurs at somewhere between 0.1 and 10.

#### Part F

Here is the list of predicted authors:

- Doc no. 0 predicted author = johnson
- Doc no. 1 predicted author = kennedy
- Doc no. 2 predicted author = kennedy
- Doc no. 3 predicted author = kennedy
- Doc no. 4 predicted author = kennedy
- Doc no. 5 predicted author = johnson
- Doc no. 6 predicted author = kennedy
- Doc no. 7 predicted author = johnson
- Doc no. 8 predicted author = kennedy
- Doc no. 9 predicted author = kennedy

## Problem 2

The overall metrics for the sci-kit learn classifier are higher. This might be due to a different alpha value. Also, it may have tokenized differently than I did with pure whitespace. There was only one document that was predicted differently between the two, and since the test set is very small this small difference can have a big impact on metrics.

## Problem 3

#### Part A

Custom NB accuracy: 0.8 F1: 0.75, sklearn NB accuracy: 0.9 F1: 0.8888888888888888.

#### Part B

Both classifiers almost have identical confusion matrices, the only difference is that our custom NB misclassified a document authored by Johnson as Kennedy, everything else was predicted similarly. 