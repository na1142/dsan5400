from nb import build_dataframe

training_df, test_df = build_dataframe("data")

# Document length comparison
lengths_by_author = {0: [], 1: []}
for index in range(len(training_df)):
    author = training_df["author"][index]
    text = training_df["text"][index]
    word_count = len(text.split())
    lengths_by_author[author].append(word_count)

for author_id in [0, 1]:
    total = 0
    for length in lengths_by_author[author_id]:
        total = total + length
    average = total / len(lengths_by_author[author_id])
    print("Author", author_id, "average word count:", average)

# Most common words per author
for author_id in [0, 1]:
    word_count_dict = {}
    for index in range(len(training_df)):
        if training_df["author"][index] == author_id:
            text = training_df["text"][index]
            words = text.split()
            for word in words:
                if word in word_count_dict:
                    word_count_dict[word] = word_count_dict[word] + 1
                else:
                    word_count_dict[word] = 1

    # find the 10 most common words without using sorted() shortcuts like most_common
    top_words = []
    for i in range(10):
        best_word = None
        best_count = -1
        for word in word_count_dict:
            if word_count_dict[word] > best_count and word not in top_words:
                best_word = word
                best_count = word_count_dict[word]
        top_words.append(best_word)

    print("Author", author_id, "top 10 words:")
    for word in top_words:
        print(word, word_count_dict[word])

# Opening and closing patterns
for author_id in [0, 1]:
    print("Author", author_id, "openings:")
    for index in range(len(training_df)):
        if training_df["author"][index] == author_id:
            text = training_df["text"][index]
            print(text[:60])

    print("Author", author_id, "endings:")
    for index in range(len(training_df)):
        if training_df["author"][index] == author_id:
            text = training_df["text"][index]
            print(text[-60:])