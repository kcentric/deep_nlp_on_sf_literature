# Note to self: "ship" etc. is not even coming in the LDA model? bcz not part of a topic?
#               Why not examine a specific novel like Robots? Or Vaughn Heppner's stuff?
# Search interstellar and "ill"

from Archive.LDA.old_LDA import LDA
import pandas as pd
from time import time

csv_file_path = "../sci-fi_text_cleaned.csv"

start_time = time()
df = pd.read_csv(csv_file_path)
cleaned_sentences = df['Cleaned sentences'].values.tolist()  # Note: this gives us a list of the cleaned sentences,
                                                             # but each sentence is not tokenized in itself. So I need
                                                             # to tokenize each now (I missed this and got a TypeError
                                                             # from gensim earlier).

# Note: from debugging, apparently the following line seems to iterate over some float values in the cleaned sentences list
# sentence_stream = [sent.split(" ") for sent in cleaned_sentences]

# So I first must convert everything in my cleaned sentences list to strings:
cleaned_sentences = [str(sentence) for sentence in cleaned_sentences]
# And now:
sentence_stream = [sent.split(" ") for sent in cleaned_sentences]  # Yayy! After this it works!

# print(sentence_stream[:30])

# First ran this line:
# lda_model = LDA(clean_text=sentence_stream[:20000])
# But received "An attempt has been made to start a new process before the
#         current process has finished its bootstrapping phase" Runtime Error.
#         Will now need to put if __name__ == '__main__' specifically
if __name__ == '__main__':
    print("Number of sentences to parse:", len(sentence_stream))
    lda_model = LDA(clean_text=sentence_stream[:10000])
    # Note: LOL! I was constantly NOT getting the right kind of topics even after I fixed the bigrams/trigrams
    #       func over in LDA.py. Why? Because I didn't fix the line where the model takes in the OG doc rather
    #       than the trigrams to make its topics. LOL!
    end_time = time()
    print(lda_model)
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time)
    if input("Wanna visualize? Enter 'Y' if yes; press Enter for no ") == "Y":
        lda_model.visualize_model()
        # Note: The order of the topics as printed in the console may not be the same
        #       as the order of the topics in the visualization. For example, the
        #       words represented in the console output in topic 2 may belong to
        #       topic 4 on the visualization. This is likely because the visualization
        #       is concerned mainly with representing topics based on how much area of the
        #       text is covered by the topic.

