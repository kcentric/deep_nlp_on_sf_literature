import pandas as pd
import time
import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer  # Could also use sent_tokenizer if unbothered about delimiters, or regex library
# import spacy  # Didn't end up using spaCy "itself" because spacy.load() wouldn't work
import en_core_web_sm  # The spacy English pipeline itself, optimized for CPU
import itertools

# nltk.download()  # Run this line whenever project is to be executed for the first time. Triggers nltk data-packs to download
#                  # to local system.

text_as_a_wordlist = []  # A global variable to make further processing convenient AFTER the corpus we're working with has
                         # been turned into a list of words once
corpus_as_a_booklist = []  # A global variable that'll usually be filled after the previous one is; from having the text as
                           # an undifferentiated wordlist, we'll divide it into books and turn it into a list of books with
                           # each book being a list of words.
                           # This gives us a list of lists or "documents" which we can then use for LDA modeling.

# Load the spacy pipeline
nlp = en_core_web_sm.load()
tokenizer = RegexpTokenizer(r"[^.?!,;]+")


def sent_tokenize(text):
    """Splits text into sentences and remove delimiters like '?' or '.' or '!' etc."""
    # Note: This function splits the text into "tokens" technically, rather than "sentences".
    #       This is effectively the same as splitting into sentences as far as analysis is concerned, because
    #       tokens are the "meaningful" units of the text.

    #return list(map(str.strip, tokenizer.tokenize(text)))  # Commented out because don't we need to strip for our use-case;
                                                            # the clean_string function (defined later) handles that.
    return list(map(str.lower, tokenizer.tokenize(text)))


def find_distances_between_instances(array, word):
    """Returns a list that contains the distances between each instance of the word, in order. The first value in the
    output list will be the distance between the FIRST and SECOND instances of the word: NOT the distance from the
    BEGINNING of the text."""

    if word not in array:
        return []

    our_iterable = [elem for elem in array]  # make a separate iterable as a copy of the input array so we don't mess
                                             # that one up
    first_index = our_iterable.index(word)  # assign the index of the first instance of our value in the list.
                                            # This is because from here we can find the distance between this and the
                                            # next occurrence of our value, and so on.
    our_iterable[first_index] += "a"  # change the element, so that this element is no longer the first instance of our
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


def fetch_pos_tags(sentences_list):
    """Fetches the POS (Parts of Speech) tags of a text that has already been split into a list of sentences. Run time
     goes up significantly as text length crosses 1000000 characters."""

    # Note 1: I set up the function to take as input a list of sentences rather than a full text itself, because
    #         the create_dataframe method, defined later on, creates a list of sentences anyway. I wished to
    #         reduce processing time as much as practical.

    # Note 2: This function has an O(n^2) time complexity, so IF it gets used, it becomes a major time-hog in our overall project
    #         code. HOWEVER, it DOES NOT HAVE to be used, IF we (or "the user") are to:
    #         a.) Select Stemming or SpaCy as the cleaning method in our main script later.
    #         b.) Still use lemmatizing, BUT pass in find_pos = False into the clean_string function.

    # nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')  # Would use this code (+more) if we took text as input
                                                                        # and wanted to split it into sentences first
    nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
    # tokens = [nltk_tokenizer.tokenize(sentence) for sentence in sentences_list]
    tokens = [nltk_tokenizer.tokenize(' '.join(sent_tokenize(sentence))) for sentence in sentences_list]

    # Note: This code was tested two times to make sure it works fully as intended
    list_of_token_lists = [[token for token in token_list] for token_list in tokens]
    pos_tokens = [nltk.pos_tag(token_list) for token_list in list_of_token_lists]  # Each nltk.pos_tag call returns a list of
                                                                                   # the tagged tokens, making this a list-within-list.
                                                                                   # nltk.pos_tag takes any sequence as input,
                                                                                   # including lists and strings
    return pos_tokens  # The final output is a list of lists, each nested list representing a sentence which we dissected,
                       # thus keeping our data well-organized.


def get_wordnet_pos(treebank_tag):
    """Returns the treebank tag's WordNet POS compliance."""
    # Note: Refer to WordNet documentation to understand fully.

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Because default Part of Speech (POS) in lemmatization is Noun:
        return wordnet.NOUN


def clean_string(text, only_remove_line_breaks=False, pos_tokens_if_lemmatizing=None, find_pos=False, stem="None"):
    """Takes in a string and removes line breaks, punctuation, stop-words, numbers, and proceeds to stem/lemmatize.
    Returns the "cleaned" text finally. Capable of nuances depending on inputs."""

    # result_str = ""

    # Lowercase
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', ' ', text)
    if only_remove_line_breaks:  # allows the user to just return the text with line breaks removed
        return text

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop-words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if word not in useless_words]  # Note: used code "not word in useless_words" earlier

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem, Lemmatize or use Spacy
    if stem == 'Stem':  # Warning: Stemming produces meaningless root words more frequently than lemmatizing
        stemmer = PorterStemmer()  # use PorterStemmer to reduce invalid outcomes as compared to other stemming methods
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem' and find_pos is True:  # Lemmatizing determines word's Part of Speech before reducing to meaningful root form.
                         # Caution: The WordNet Lemmatizer may still produce inaccurate results if a POS tag isn't passed in for each word,
                         #          because it then treats every word as a noun. I needed to create a separate function to automate POS fetching
                         #          for each word, but it increased the processing time by an order of magnitude when used.

        lem = WordNetLemmatizer()
        pos_tokens = pos_tokens_if_lemmatizing

        # See if this code works?
        # pos_tokens is the list of POS tokens for the sentence we're working on right now, so it is NOT a list of lists
        #   but rather a list of tuples. So, we can refer to the word and its POS tag to pass into the lemmatizer and
        #   get a list of properly lemmatized words as a result.
        text_stemmed = [lem.lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos_tokens]

    elif stem == 'Lem':  # Lemmatizing without finding POS tags; MUCH faster (linear time complexity) than with POS tags
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]

    # When the user wants to be most thorough if slower than stemming or non-POS lemmatizing, Spacy should be used
    # However, note: Spacy's max text length per-processing is 1*10^6 characters, to prevent memory allocation errors.
    # Therefore, to prevent hassle, I used lemmatizing (without finding POS tags).
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    result_str = ' '.join(text_stemmed)

    return result_str

# print(clean_string("Hello. My name is Krishna; I am the maker of this project."
#                    "vvv the a an,", only_remove_line_breaks=True))


def create_dataframe(text, cleaning_method, find_pos=False):
    """Outputs a pandas dataframe from a text, with one column containing sentences, the other column containing the cleaned
     sentence for analysis. For input, takes in the text and the NLP cleaning method to use: Stemming, Lemmatizing,
     or the Spacy model.

     During processing, also creates a representation of the (CLEANED) corpus as a list of words and assigns it to a
     global variable for further usage."""

    # Note: find_pos is set to False by default, to prevent innocent lemmatizing from leading into a crashingly-long processing time
    # for large text files - like the one used in this project (57 MB size). Finding POS gives the program a O(n^2)+ time complexity.
    # Without that, we remain with O(n) to O(log(n)) time complexity, which is preferable. Note than sentence-cleaning only remains
    # in this range based on the fact that each sentence does not have a very large size approaching n. Otherwise technically,
    # sentence-cleaning can go up to O(n^2) itself.

    list_of_sentences = sent_tokenize(clean_string(text, only_remove_line_breaks=True))  # lists the text's sentences in order

    if cleaning_method == "Lem" and find_pos:
        pos_tokens = fetch_pos_tags(list_of_sentences)
        # Following line includes code to pass the POS tokens of each sentence along with the sentence into our lemmatizer
        cleaned_sentences = [clean_string(sentence, pos_tokens_if_lemmatizing=pos_tokens[list_of_sentences.index(sentence)], stem=cleaning_method) for sentence in list_of_sentences]
    else:
        cleaned_sentences = [clean_string(sentence, stem=cleaning_method) for sentence in list_of_sentences]

    # Note: This line creates a slight inefficiency by adding another O(n) operation; however it's "alright"
    list_of_lists = [sentence.split() for sentence in cleaned_sentences]  # List of CLEANED sentences as a list of lists

    # The GLOBAL variable
    global text_as_a_wordlist
    text_as_a_wordlist = list(itertools.chain.from_iterable(list_of_lists))  # Now we can easily use this wordlist for further processing
    print("List of sentences:", list_of_sentences[0:50])
    print("Cleaned sentences:", cleaned_sentences[0:50])
    print("List of words:", text_as_a_wordlist[0:100])
    # print(find_distances_between_instances(text_as_a_wordlist, "copyright"))
    data_dict = {
      "Sentences": list_of_sentences,
      "Cleaned sentences": cleaned_sentences,
    }
    dataframe = pd.DataFrame(data_dict)
    return dataframe


def split_into_books(corpus_as_wordlist=None):  # Could also make this func more general by taking in the word to split by as an argument
    """Built specifically to suit this project. Takes in the corpus as a list of words and attempts to separate it into
    "books" based on the appearance of "copyright" at suitable intervals. Outputs several print statements during the
     process to give user a sense of whether the work is being done right. Returns the corpus as a list of lists, each
     list containing one "book"-unit that's been ascertained by this method. Note, as this particular overall project code
     is structured, the return value contains a CLEANED version of the corpus (ready for further NLP processing)."""

    # Note: The function attempts to ascertain whether the word "copyright", in a particular instance of its appearance,
    #       isn't appearing merely as part of a story (for example, if a character is talking about her copyright getting
    #       violated) or as part of a preface where an editor is talking about someone's copyright page (I visually
    #       spotted an instance of this in the data)
    #       hasn't appeared just as part of a fictional work (for example, there are instances in

    # Note: I had a thought about the function to include web scraping in it in some way; possibly to verify the
    #       book-hood of the book-units it extracts. However, it did not seem necessary.

    global text_as_a_wordlist
    if corpus_as_wordlist is None:  # Maybe also check if the input is not a list?
        corpus_as_wordlist = text_as_a_wordlist

    corpus = corpus_as_wordlist
    copyright_v1_instances = corpus.count('copyright')
    # copyright_v2_instances = corpus.count(' copyright ')  # This extra test-line was for when the function was originally
                                                            # designed to operate on a string argument. Now that we have
                                                            # the text as a cleaned list, there won't be any spaces around
                                                            # any of the words.
    print(f"The word 'Copyright' appears {copyright_v1_instances} times.\n")

    # Find distances between each copyright instance
    distances = find_distances_between_instances(corpus, 'copyright')
    # Print so user can see whether we're on the right path
    print("List of distance b/w copyright instances (first 100 only):", distances[0:100])
    print(f"The length of this list is {len(distances)}, which matches with the number of copyright instances (dist "
          f"between beginning of text and first copyright instances was not measured, so we should have 'n-1' here).")
    print(f"Also, the distances between each copyright instance seems reasonable: tens of thousands of words at minimum"
          f" is expected.")

    # Find smallest distance between two copyrights. I know already that at least once, the word "copyright" is used
    # merely in a discussion by the editor
    min_dist = 0
    max_dist = 0
    if len(distances) > 2:
        min_dist = min(distances)
        max_dist = max(distances)

    print(f"The smallest distance between two copyrights is: {min_dist}")
    print(f"The largest distance between two copyrights is: {max_dist}")

    # Since this value is too small for it to be a separate magazine/book, we remove that copyright instance and will not
    # split around it


    print("Proceeding to split ...")
    # Using itertools
    split_corpus = [list(group) for k, group in itertools.groupby(corpus, lambda x: x == 'copyright') if not k]
    # print(split_corpus[0:2])
    print("Length of split corpus (no. of books):", len(split_corpus))

    return split_corpus

def write_list_to_file(wordlist_or_booklist=None):
    """Intended to write the list contained in one of the GLOBAL variables (text_as_a_wordlist or corpus_as_a_booklist)
    to a txt file. If no argument is passed, defaults to writing text_as_a_wordlist (whole corpus) to the file.

    Helps to give us a PROCESSED version of the text suitable for LDA modeling."""
    # Note: What is the difference between just the original text of our corpus and this?
    #       Answer: The original corpus was "unclean" and not in a format ready for modeling.
    #               Now we shall have it as a list of lists, which each list more or less containing
    #               one book.
    if wordlist_or_booklist is None:
        global text_as_a_wordlist
        wordlist_or_booklist = text_as_a_wordlist

    with open("corpus_as_cleaned_wordlist.txt", "w+") as our_file:
        our_file.writelines(wordlist_or_booklist)

# Testing script using a small sample text file
test_filepath = "small_sample_text_for_testing.txt"
test_text = open(test_filepath, "r").read().lower()
start_time = time.time()
cleaned_test_text = create_dataframe(test_text, "Lem", find_pos=True)  # adjust appropriately to use other cleaning methods
end_time = time.time()
elapsed_time = end_time - start_time
new_file = open("small_sample_text_cleaned.csv", "w+")
cleaned_test_text.to_csv("small_sample_text_cleaned.csv")
print("Time taken for cleaning and organizing SAMPLE TEXT:", elapsed_time, "s")
print(f"Pandas Dataframe:\n{cleaned_test_text}\n")
print(text_as_a_wordlist)
split_into_books(text_as_a_wordlist)

## Comment-out the following block of main-script code and run the file to perform sample testing. Uncomment back in when done.

## Main Script

database_filepath = '/Users/anuragtripathi/Downloads/internet_archive_scifi_v3.txt'
original_text = open(database_filepath, "r").read().lower()  # Text length: 94040093 characters

# print(original_text.count("Copyright"))  # very interesting how this line wasn't working :-)
#                                            because the text has been converted into lowercase. "C" is nowhere to be found.

print(f"## Trying to ascertain number of distinct 'books' in the corpus:\n\n"
      f"Number of times 'copyright' appears: {original_text.count('copyright')}")

print("\n\n## Creating dataframe from text. Please wait.")
# Just to measure time
start_time = time.time()
# split_into_books(original_text)

# Clean the text by splitting it into sentences and creating a dataframe with the sentences and their cleaned versions
clean_text = create_dataframe(original_text, "Lem")  # Note: for a 57 MB input text file, processing time takes above 270 seconds, approx
write_list_to_file()
# Write the dataframe to a csv to that we don't need to take time for cleaning again
# file = open('sci-fi_text_cleaned.csv', "w+")
# clean_text.to_csv('sci-fi_text_cleaned.csv')

end_time = time.time()

elapsed_time = end_time - start_time

print("Time taken for cleaning and organizing:", elapsed_time, "s")
print(f"Pandas Dataframe:\n{clean_text}")
print(split_into_books(text_as_a_wordlist)[0:1])
## Comment out the above main-script code and run the file to perform sample testing. Uncomment back in when done.
