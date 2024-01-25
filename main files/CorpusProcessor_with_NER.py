# NOTE: SEE TODO s below
# TODO: Change create_dataframe function to take in a data_dict and turn it into a dataframe. make var called NER_dataframe
# TODO: Fix TEST() method
# NOTE: another TODO, change all the file names to produce files in correct directories
# TODO: Check if my adjustment to the reading methods keeps stuff functional still!
#  Run this file once as Main, for pre-final testing. Had adjusted the initializtion procedure to
#  do "initial_prep" rather than "create_dataframe" by default, to reduce time for init.

from NER import NER
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
                   # to local system.

FILEPATH = '../Input Files/internet_archive_scifi_v3.txt''

# Path to a sample corpus
FILEPATH = '../Input Files/small_sample_text_for_testing.txt' # DELETE (or comment-out) this line in order to process the "real" corpus

# Functions to help the class
def find_distances_between_instances(array, word):
    """Takes in an array of words, and a word. Returns a list that contains the distances between each instance of the word,
    in order. Note: the first value of the output list will be the distance between the FIRST and SECOND instances of the word:
    NOT the distance from the BEGINNING of the text."""

    if word not in array:
        return []

    our_iterable = [elem for elem in array]  # make a separate iterable as a copy of the input array, so we don't mess
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


def get_wordnet_pos(treebank_tag):
    """Takes in a part-of-speech "treebank" tag. Returns the tag's WordNet POS compliance by referring to *nltk*'s
    *wordnet* module."""
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


def remove_excess_instances(corpus, distances_between_instances, word, min_acceptable_distance=6000):
    # TODO: COMPLETE THIS METHOD
    """Takes in the text as a wordlist, with the distances between each instance of the word we're splitting it by,
    and the word itself. Removes excess instances of the word by iteratively finding excessively small distances
    between any two instances: biased towards larger distances. NOT FOOLPROOF.

    *min_acceptable_distance* defaults to 10000, assuming that if we're dealing with a corpus of books, each is at
    least 10000 words long."""
    edited_corpus = []
    # Find smallest distance between two instances of word. For eg, I know the word "copyright" is oft
    # is used merely in a discussion by an editor rather than as an actual copyright declaration
    min_dist = 0
    max_dist = 0
    if len(distances_between_instances) > 2:
        min_dist = min(distances_between_instances)
        max_dist = max(distances_between_instances)

    print(f"The smallest distance between two copyrights is: {min_dist}")
    print(f"The largest distance between two copyrights is: {max_dist}")

    # Since this value is too small for it to be a separate magazine/book, we remove that copyright instance and will not
    # split around it

    return edited_corpus


class CorpusProcessor:
    """Initializes with a filepath. Contains a complete set of methods to process a text file, specifically designed to
    ready the data for LDA modeling but helpful for many other kinds of tasks also. Capable of saving output data to
    txt files (and/or csv files for a Pandas dataframe).

    Note: *corpus_contains_mostly_stories*, if set to True, will lead the model to cleaning out words like "said" and
    "replied" so that non-meaningful dialogue tags don't contribute to noise in LDA modeling."""

    def __init__(self, filepath=FILEPATH, cleaning_method="Lem", measure_processing_time=True,
                 corpus_contains_mostly_stories=True):  # For user convenience, default to measuring
                                                        # processing time.
        self.list_of_sentences_original = []  # Hold the original text as a list of sentences
        self.list_of_sentences_cleaned = []  # Hold initially-processed text as a list of sentences
        self.dataframe = None  # To hold a Pandas dataframe of sentences and cleaned sentences from our text

        self.text_as_a_wordlist = []  # To make further processing convenient AFTER the corpus we're working with has
                                      # been turned into a list of words once.
        self.corpus_as_a_booklist = []  # Will usually be filled after the previous one is; from having the text as
                                        # an undifferentiated wordlist, we'll divide it into books and turn it into a list of books
                                        # with each book being a list of words.
                                        # This gives us a list of lists or "documents" which we can then use for LDA modeling.

        # Load the spacy pipeline
        self.nlp = en_core_web_sm.load()
        self.tokenizer = RegexpTokenizer(r"[^.?!,;]+")

        start_time = time.time()  # Record time

        # Run the basics. Note: The initial_prep method is meant to do a lot of essential set-up for the other methods
        #                       by executing first
        self.process(filepath, cleaning_method, corpus_contains_mostly_stories)

        if measure_processing_time is True:
            print("Time taken for initial processing:", time.time() - start_time, "seconds")
            print("Ready for further processing.")

    def process(self, filepath, cleaning_method, corpus_contains_mostly_stories=True):
        """Open the file and perform initial cleaning."""
        with open(filepath, "r") as file:
            original_text = file.read().lower()
            self.initial_prep(original_text, cleaning_method,
                              corpus_contains_mostly_stories=corpus_contains_mostly_stories)
            #self.dataframe = self.create_dataframe(original_text, cleaning_method, corpus_contains_mostly_stories)

    def sent_tokenize(self, text):
        """Split text into sentences and remove delimiters like '?' or '.' or '!' etc."""

        # Note: This function splits the text into "tokens" technically, rather than "sentences".
        #       This is effectively the same as splitting into sentences as far as analysis is concerned, because
        #       tokens are the "meaningful" units of the text.

        return list(map(str.lower, self.tokenizer.tokenize(text)))

    def fetch_pos_tags(self, sentences_list):
        """Fetches the POS (Parts of Speech) tags of a text that has already been split into a list of sentences. Runtime
         goes up significantly if text length crosses 1000000 characters."""

        # Note 1: I set up the method to take as input a list of sentences rather than a full text itself, because
        #         the create_dataframe method, defined later on, creates a list of sentences anyway. I wished to
        #         reduce processing time as much as practical.

        # Note 2: This method has an O(n^2) time complexity, so IF it gets used, it becomes a major time-hog in our overall project
        #         code. HOWEVER, it DOES NOT HAVE to be used, IF we (or "the user") are to:
        #         a.) Select Stemming or SpaCy as the cleaning method in our main script later.
        #         b.) Still use lemmatizing, BUT pass in find_pos = False into the clean_string function.

        # nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')  # Would need this code (and more) if we took text as input
                                                                             # and wanted to split it into sentences first
        nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        # tokens = [nltk_tokenizer.tokenize(sentence) for sentence in sentences_list]
        tokens = [nltk_tokenizer.tokenize(' '.join(self.sent_tokenize(sentence))) for sentence in sentences_list]

        # Note: This code was tested two times to make sure it works fully as intended
        list_of_token_lists = [[token for token in token_list] for token_list in tokens]
        pos_tokens = [nltk.pos_tag(token_list) for token_list in list_of_token_lists]  # Each nltk.pos_tag call returns a list of
                                                                                       # the tagged tokens, making this a list-within-list.
                                                                                       # nltk.pos_tag takes any sequence as input,
                                                                                       # including lists and strings.
        return pos_tokens  # The final output is a list of lists, each nested list representing a sentence which we dissected,
                           # thus keeping our data well-organized.

    def clean_string(self, text, only_remove_line_breaks=False, pos_tokens_if_lemmatizing=None, find_pos=False,
                     stem="None", working_on_stories=True):
        """Takes in a string and removes line breaks, punctuation, stop-words, numbers, and proceeds to stem/lemmatize.
        Returns the "cleaned" text finally. Capable of nuances depending on inputs.

        Note: *find_pos* parameter (find Parts of Speech) defaults to False because POS-fetching increases processing time
        by orders of magnitude."""

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
        if working_on_stories:
            useless_words = useless_words + ['said', 'replied']

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
            text_stemmed = [lem.lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos_tokens]

        elif stem == 'Lem':  # Lemmatizing without finding POS tags; MUCH faster (linear time complexity) than with POS tags
            lem = WordNetLemmatizer()
            text_stemmed = [lem.lemmatize(y) for y in text_filtered]

        # When the user wants to be most thorough if slower than stemming or non-POS lemmatizing, Spacy should be used
        # However, note: Spacy's max text length per-processing is 1*10^6 characters, to prevent memory allocation errors.
        # Therefore, to prevent hassle, it's good to use normal lemmatizing mostly (without finding POS tags).
        elif stem == 'Spacy':
            text_filtered = self.nlp(' '.join(text_filtered))
            text_stemmed = [y.lemma_ for y in text_filtered]
        else:
            text_stemmed = text_filtered

        result_str = ' '.join(text_stemmed)

        return result_str

    def initial_prep(self, text, cleaning_method, find_pos=False, corpus_contains_mostly_stories=False):
        """For input, mainly requires the text and the NLP cleaning method to use: Stemming, Lemmatizing,
         or the Spacy model. Extracts sentences, cleans them, and uses them to also create a representation of the
         (CLEANED) corpus as a list of words. Then assigns this list of words to an object attribute also."""

        # Note: find_pos is set to False by default, to prevent innocent lemmatizing from leading into a crashingly-long processing time
        # for large text files - like the one used in this project (57 MB size). Finding POS gives the program a O(n^2)+ time complexity.
        # Without that, we remain with O(n) to O(log(n)) time complexity, which is preferable. Note than sentence-cleaning only remains
        # in this range based on the fact that each sentence does not have a very large size approaching n. Otherwise technically,
        # sentence-cleaning can go up to O(n^2) itself.

        list_of_sentences = self.sent_tokenize(self.clean_string(text, only_remove_line_breaks=True))  # lists the text's sentences in order

        if cleaning_method == "Lem" and find_pos:
            pos_tokens = self.fetch_pos_tags(list_of_sentences)
            # Following line includes code to pass the POS tokens of each sentence along with the sentence into our lemmatizer
            cleaned_sentences = [self.clean_string(sentence, pos_tokens_if_lemmatizing=pos_tokens[list_of_sentences.index(sentence)],
                                                   stem=cleaning_method, working_on_stories=corpus_contains_mostly_stories)
                                 for sentence in list_of_sentences]
        else:
            cleaned_sentences = [self.clean_string(sentence, stem=cleaning_method,
                                                   working_on_stories=corpus_contains_mostly_stories) for sentence in
                                 list_of_sentences]

        # Note: This line creates a slight inefficiency by adding another O(n) operation; however it's "alright"
        list_of_lists = [sentence.split() for sentence in cleaned_sentences]  # List of CLEANED sentences as a list of lists

        self.text_as_a_wordlist = list(itertools.chain.from_iterable(list_of_lists))  # Now we can easily use this wordlist
                                                                                      # for further processing
        self.list_of_sentences_original = list_of_sentences
        self.list_of_sentences_cleaned = cleaned_sentences
        print("List of sentences (partial):", list_of_sentences[0:50])
        print("Cleaned sentences (partial):", cleaned_sentences[0:50])
        print("List of words (partial):", self.text_as_a_wordlist[0:100])

        # Having some fun :)
        return "This function does not return prepped text, but rather just preps the text to now be contained" \
               "as the CorpusProcessor's attributes. Please call those attributes if you wish to see the cleaned text :-)"

    def create_dataframe(self, write_to_csv=True):
        """Assigns to CorpusProcessor object's *dataframe* attribute a Pandas dataframe representing sentences and
        cleaned sentences. Also returns the dataframe for user's convenience. If *write_to_csv* is True, writes the
        dataframe to a csv file also."""

        data_dict = {
          "Sentences": self.list_of_sentences_original,
          "Cleaned sentences": self.list_of_sentences_cleaned,
        }
        dataframe = pd.DataFrame(data_dict)
        self.dataframe = dataframe
        if write_to_csv:
            self.write_dataframe_to_csv("../sci-fi_text_cleaned.csv")
        return dataframe  # Assigns dataframe to object attribute as well as returns it, for user convenience


    def split_into_books(self, corpus_as_wordlist=None, split_by="copyright", squint_at_results=False):
        """Takes in a corpus as a list of words and attempts to separate it into "books" based on the appearance of
        the word passed as argument to *split_by* at suitable intervals. *split_by* defaults to "copyright".

        Also outputs several print statements during the process to give user a sense of whether the work is being done right.

        Returns the corpus as a list of lists, each list containing one "book"-unit that's been ascertained by this method.
        Note: the return value is typically meant to contain a CLEANED version of the corpus (ready for further NLP processing),
        particularly LDA topic-modeling.

        Note: If *squint_at_results* is set to True, method attempts to ascertain whether the *split_by* word, in a particular
        instance of its appearance, isn't appearing merely as part of a story (for example, if a character is talking about her
        copyright getting violated) or as part of a preface where an editor is talking about someone's copyright page (I visually
        spotted an instance of this in the data). *squint_at_results* defaults to False because:

        1.) The means I ascertain false results by is not foolproof, because it's fairly simplistic. It usually works well with "copyright"
        as *split-by*, but with other words it could spectacularly fail.

        2.) Using LDA topic modeling on a large enough corpus, it doesn't make a significant difference to have one or two "falsely split"
        books."""

        # Note: I had a thought about this method to include web scraping in it in some way; possibly to verify the
        #       book-hood of the book-units it extracts. However, it did not seem necessary.

        if corpus_as_wordlist is None:  # Could also check if the input is not a list
            corpus_as_wordlist = self.text_as_a_wordlist

        corpus = corpus_as_wordlist
        copyright_v1_instances = corpus.count(split_by)
        print(f"The word '{split_by}' appears {copyright_v1_instances} times.\n")

        # Find distances between each copyright/word instance
        distances = find_distances_between_instances(corpus, split_by)
        # Print so user can see whether we're on the right path
        print(f"List of distance b/w '{split_by}' instances (first 100 only):", distances[0:100])
        print(f"The length of this list is {len(distances)}, which matches with the number of '{split_by}' instances (dist "
              f"between beginning of text and first '{split_by}' instance was not measured, so we should have 'n-1' here).")

        # Thought: Could also add a couple lines here to estimate how many avg words there should be per book/document
        #          based on total number of books/docs and avg length of each book/doc.
        print(f"Also, if you are splitting a corpus of books, the distances between each '{split_by}' instance above is"
              f" reasonable if it's on avg tens of thousands of words at minimum. That's expected.")

        if squint_at_results is True:
            # Note: If using a different word than "copyright" and something other than a corpus of full-scale books,
            #       you need to pass a fourth argument, containing the minimum acceptable distance between two instances
            #       at which you'd accept that the text in between was a distinct document.
            corpus = remove_excess_instances(corpus, distances, split_by)

        print("Proceeding to split ...")
        # Using itertools
        split_corpus = [list(group) for k, group in itertools.groupby(corpus, lambda x: x == split_by) if not k]
        print("Length of split corpus (no. of books):", len(split_corpus))
        print("First two elements of split corpus:", split_corpus[0:2])

        self.corpus_as_a_booklist = split_corpus

        return split_corpus

    def extract_named_entities(self, text, return_cleaned_text=True):
        """Perform NER using Spacy on a text. Assign the named entities as a list to attribute *named_entities*, and
        write them to a file. Also, if *return_cleaned_text* is True, remove named entities from the text and return
        the resulting text. Assign cleaned text to *text_without_NER* attribute."""
        pass

    def write_dataframe_to_csv(self, filepath):
        with open(filepath, "w+") as our_file:
            self.dataframe.to_csv(our_file)

    def write_lists_to_file(self, wordlist_or_booklist="", write_both=True):
        # TODO: Adjust for write_both parameter. Or just remove it and always write both.
        """Intended to write the list contained in one of the main instance attributes (*text_as_a_wordlist* or
        *corpus_as_a_booklist*) to a txt file. If no argument is passed, defaults to writing *text_as_a_wordlist*
        (whole corpus) to the file.

        *write_both* parameter defaults to True. Method defaults to writing *text_as_a_wordlist* AND
        *corpus_as_a_booklist* to separate files. If False, we only write *corpus_as_a_booklist*.

        Helps to give us a PROCESSED version of the text suitable for LDA modeling."""
        # Note: Question: What is the difference between just the original text of our corpus and this?
        #       Answer: The original corpus was "unclean" and not in a format ready for modeling.
        #               Now we shall have it as a list of lists, with each list more or less containing
        #               one book.
        if wordlist_or_booklist == "":
            wordlist_or_booklist = self.text_as_a_wordlist

        with open("../Data Files (Readable)/corpus_as_cleaned_wordlist.txt", "w+") as our_file:
            # Make sure that when we write to a file we'll have a readable string with spaces, per book. Not a letter-stream!
            wordlist_or_booklist = [" ".join(list_of_words) + "\n" for list_of_words in wordlist_or_booklist]

            print(len(wordlist_or_booklist), "\nFirst 2 elements:", wordlist_or_booklist[0:2])

            # Write each book to one line in the file. Note that this will make for very long lines. However,
            # IDEs (Pycharm at least) provide auto-enabled reading features that make it readable regardless.
            our_file.writelines(wordlist_or_booklist)

    def read_corpus_from_file(self, filepath_1="corpus_as_cleaned_wordlist.txt", filepath_2="text_as_a_cleaned_wordlist.txt"):
        with open(filepath_1, "r+") as our_file:
            self.corpus_as_a_booklist = [book.split(" ") for book in our_file.readlines()]

        with open(filepath_2, "r+") as our_file:
            text_as_a_wordlist = our_file.readlines()[0]    # There's only one element in our readlines() output, and
                                                            # that's the one we want.
            self.text_as_a_wordlist = [text_as_a_wordlist.split(" ")]

    def read_dataframe_from_csv(self, filepath="sci-fi_text_cleaned.csv"):
        self.dataframe = pd.read_csv(filepath)

    def test(self, filepath="small_sample_text_for_testing.txt", cleaning_method="Lem", find_pos=False):
        """Test the processor on a small default sample file, or a file of user's choice - in that case,
        filepath must be specified. Default *cleaning_method* is simple Lemmatization (input "Stem" or "Spacy" to change).

        Note: This method DOES NOT perform corpus-splitting (*split_into_books*), NOR does it write those lists to a file.
        """

        # Because the default sample file is quite small, we can afford the time-complexity increase of POS-fetching
        if filepath == "small_sample_text_for_testing.txt":
            find_pos = True

        test_text = open(filepath, "r").read().lower()

        # Will test time taken also
        start_time = time.time()

        # Create Pandas dataframe and in the process generate values for some of our needed instance attributes also
        cleaned_text_as_df = self.create_dataframe()

        # Calculate time
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open("small_sample_text_cleaned.csv", "w+") as new_file:  # will create the file if it doesn't exist
            cleaned_text_as_df.to_csv("small_sample_text_cleaned.csv")

        # Print statements
        print("Time taken for cleaning and organizing TEST text:", elapsed_time, "s")
        print(f"Pandas Dataframe (first 30 rows):\n{cleaned_text_as_df.head(10)}\n")
        print(self.__str__())

    def __str__(self):
        return f"Text as a wordlist (up to first 5000 elements only):\n{self.text_as_a_wordlist[:5000]}" \
               f"\nCorpus as a list of documents, with each document as a wordlist (first two documents only):\n" \
               f"{self.corpus_as_a_booklist[0:2]}"


if __name__ == '__main__':
    processor = CorpusProcessor()
    processor.create_dataframe()
