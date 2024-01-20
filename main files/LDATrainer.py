# TODO: BIG NOTE!! REMOVE STRINGS SHORTER THAN SEVERAL 1000 CHARACTERS TO REMOVE FALSELY SPLIT BOOKS!
# TODO: Modify so that IF files exist (or a read from file parameter is set to True), we just read info from a file
# TODO: Adjust so that we can pass in no. of LDA training passes from HERE
# TODO: now do seeding ...
#  After this file runs once, I can then use the disk image of the model from saved file
# TODO: Record! Wait a minute ... I don't need a text processor object if I have the list of books, LOL!!!
#  Didn't need to waste time on that ...
# TODO: fix everything now. Somehow, the text file is not being read??


# Trigrams use kro, not quadgrams
# now question is, what does it mean when some words appear together as a topic. how to interpret it.

from LDA import LDA
# from CorpusProcessor import CorpusProcessor
from time import time
from CorpusProcessor_with_NER import CorpusProcessor

# Path to SF corpus
FILEPATH = '../Data Files (Readable)/Input Files/internet_archive_scifi_v3.txt'

# Path to test text. Uncomment and run to test model's basic functionality quickly
# FILEPATH = 'Data Files (Readable)/Input Files/small_sample_text_for_testing.txt'

class TrainedLDA:
    """Parameter *number_of_documents* takes in the number of books/documents to train the model on, from the corpus.
     More documents builds greater accuracy, but increases training time also. Generates a trained LDA model.

     Note: In order to load texts from a file, you must set *load_texts_from_file* parameter as True during initialization.
     """
    def __init__(self, number_of_documents=2, load_texts_from_file=False,
                 corpus_as_booklist_filepath='Data Files (Readable)/corpus_as_cleaned_wordlist.txt'):

        self.text_processor = "No Text Processor was initialized"
        self.list_of_books = []  # To hold the list of books, each book being a list of words

        # Text preparation
        if load_texts_from_file is True:  # DRASTICALLY reduces overall runtime
            with open(corpus_as_booklist_filepath, "r+") as file:
                books_as_strings = file.readlines()
                self.list_of_books = [book[:-1].split(" ") for book in books_as_strings]

                # For user to see
                print("Please verify:")
                print("First two books:", self.list_of_books[0:2])
                print("Number of books:", len(self.list_of_books))
                print("The last character of book 1, to see if newline was indeed what we removed (yes, "
                      "you'll just see an actual newline in your console):",
                      books_as_strings[0][-1])
        else:
            self.text_processor = CorpusProcessor(filepath=FILEPATH)
            self.prepare_text()

        # Timekeeping
        self.start_time = time()

        # Train the LDA model
        self.model = LDA(clean_text=self.list_of_books[0:100], num_topics=2, num_passes=1)

        # Timekeeping
        self.end_time = time()

        # Print statements and visualization
        self.after_training()

    def prepare_text(self):
        self.text_processor.split_into_books()
        self.text_processor.write_lists_to_file(self.text_processor.corpus_as_a_booklist)
        self.list_of_books = self.text_processor.corpus_as_a_booklist

    def after_training(self):
        print("Time taken for training:", self.end_time - self.start_time, "seconds")
        print("Text info:\n", self.text_processor)
        print("LDA info:\n", self.model)
        if input("Wanna visualize? Enter 'Y' if yes; press Enter for no ") == "Y":
            self.model.visualize_model()

if __name__ == '__main__':
    trained = TrainedLDA(load_texts_from_file=True)
