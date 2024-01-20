import nltk
import itertools
#
from nltk.tokenize import RegexpTokenizer
#
tokenizer = RegexpTokenizer(r"[^.?!,;]+")


def sent_tokenize(text):  # Split text into sentences and remove delimiters like ? or . or ! etc.
    # Note: This function splits the text into TOKENS specifically, rather than SENTENCES.
    #       This is effectively the same as splitting into sentences as far as analysis is concerned, because
    #       tokens are the "meaningful" units of the text.

    # return list(map(str.strip, tokenizer.tokenize(text)))  # Commented out because don't need to strip in our use-case; clean_string handles that.
    return list(map(str.lower, tokenizer.tokenize(text)))
#
# print(sent_tokenize("I wanted to use wordnet lemmatizer in python and I have learnt that the default pos tag is NOUN and that it does not output the correct lemma for a verb, unless the pos tag is explicitly specified as VERB."
#                     "My question is what is the best shot inorder to perform the above lemmatization accurately?"))

# nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
# token = [nltk_tokenizer.tokenize(' '.join(sent_tokenize("I am going to eat some food."
#                                                      "We uuuu be being goodness,!"))), nltk_tokenizer.tokenize(' '.join(sent_tokenize("I am going to eat some food."
#                                                      "Now U also come and let's have something healthy exA")))]
# list_of_tokens = [[token for token in token_list] for token_list in token]
# print("List of lists:", list_of_tokens)
# combined_tokens = list(itertools.chain.from_iterable(list_of_tokens)) # turn the corpus into a sequential list of words
# #print([nltk.pos_tag(token) for token in list_of_tokens])
# print("One list:", combined_tokens[0:3])


def find_distances_between_instances(array, word):  # returns a list that contains the distances between each instance
                                                    # of the word, in order
    our_iterable = array
    first_index = our_iterable.index(word)  # assign the index of the first instance of our value in the list.
                                            # This is because from here we can find the distance between this and the
                                            # next occurrence of our value, and so on.
    array[first_index] += "a"  # change the element, so that this element is no longer the first instance of our
                               # value when index() is called on the next iteration
    current_index = first_index
    distances = []
    for i in range(first_index, len(our_iterable)):  # loop iterating from the first index of our value to the end of
                                                     # the list
        elem = our_iterable[i]
        if elem == word:
            index = our_iterable.index(elem)
            distance = index - current_index  # find the distance between this instance of the element and the previous
                                              # by subtracting indices
            distances.append(distance)
            current_index = index  # now we will shift so that we can find the next distance next time
            our_iterable[i] += "a"  # change the element, so that this element is no longer the first instance of our
                                    # value when index() is called on the next iteration

    return distances


# testing distance-finding function
test_filepath = "../Data Files (Readable)/Input Files/small_sample_text_for_testing.txt"
test_text = open(test_filepath, "r").read().lower()
nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
token = [nltk_tokenizer.tokenize(' '.join(sent_tokenize(test_text)))]
list_of_tokens = [[token for token in token_list] for token_list in token]
print("\nList of lists:", list_of_tokens)
combined_tokens = list(itertools.chain.from_iterable(list_of_tokens))  # turn the corpus into a sequential list of words
print("One list:", combined_tokens)
print(find_distances_between_instances(combined_tokens, "hinduism"))

print(type(test_text))

# if type(test_text) is not <class 'str'>:
  #  print("Hi")

if type(test_text is str):
    print("Hello")

# database_filepath = '/Users/anuragtripathi/Downloads/internet_archive_scifi_v3.txt'
# file = open(database_filepath, "r").read()
#print(file[1:10000])
#print(file)

#print(nltk.Text(file))

# Created this function in future_main_file.py but it ended up in disuse.
# def split_into_books(corpus):  # Take the corpus and creatively try splitting it into 'book'-units by testing appearance
#                                 # of "copyright" with its variations as demarcator of book-beginnings. Also pull
#                                 # some text from either side of the copyright declaration to ascertain whether the word
#                                 # hasn't appeared just as part of a fictional work (for example, there are instances in
#                                 # the corpus where a story contains the mention of a "copyright" as part of the plot
#                                 # itself).
#
#     # Possible addition to function: Use web-scraping to check each of these works online,
#     # seeing if each one where a genuine-seeming copyright is mentioned isn't a 'fictional'
#     # book (a book which doesn't really exist) - as opposed to a book of fiction. Am not
#     # using such web-scraping right now because depending on how the text is organized in
#     # my available corpus versus how each book's info might pop up on the internet, I might
#     # have to do some more complex work to get accurate results.
#
#     corpus = corpus.lower()
#     copyright_v1_instances = corpus.count('copyright')
#     copyright_v2_instances = corpus.count(' copyright ')
#
#     print(f"The word 'Copyright' appears {copyright_v1_instances} times\n' Copyright ' (with a space around it) appears "
#           f"{copyright_v2_instances} times")
#
#     ## Now find distances between each copyright instance so user can visually see whether we're on the right path.
#
#     # Some similar code as we have in our find_pos_tags function
#     sentences_list = sent_tokenize(corpus)
#     nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
#
#     # The following gives us a list of lists
#     list_of_tokenized_sentences = [nltk_tokenizer.tokenize(' '.join(sent_tokenize(sentence))) for sentence in sentences_list]
#     # print("List of sentences-as-lists:", list_of_tokenized_sentences)  # Print the whole humongous list
#     print("Glimpse list of sentences-as-lists:", list_of_tokenized_sentences[0:5])
#
#     # Turning the list of lists into one big list of words - which remain in the same order as in the corpus
#     tokens = list(itertools.chain.from_iterable(list_of_tokenized_sentences))
#     # print("Corpus as a list of words:", tokens)  # Print the whole humongous list
#     print("Glimpse of corpus as a list of words:", tokens[0:50])
#
#     copyright_instances_in_list = tokens.count("copyright")
#     print(f"Verifying: In corpus-as-list too, copyright appears {copyright_instances_in_list} times.")




