# Note: DELETE nltk_data folder after project is done. It consumes 3.59 GB !!
# Note: all the "notes" under the functions can be done using ''' ''' to make them documentation.
#       Also, I could modularize the whole project by creating classes and using OOP, for improvement.

# import pathlib
import pandas as pd
import time
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer  # Could also use sent_tokenizer if unbothered about delimiters, or regex lib
# import spacy  # didn't end up using spaCy "itself" because spacy.load() wouldn't work
import en_core_web_sm  # it is the spaCy English pipeline itself, optimized for CPU

# nltk.download()  # Run this line whenever project is to be executed for the first time

# Load spacy
#nlp = spacy.load('en_core_web_sm') # Note: Install using "python -m spacy download en_core_web_sm" command on terminal

#nlp = spacy.load('./Library/Python/3.10/lib/python/site-packages/en_core_web_sm/en_core_web_sm-3.7.1')

# path = pathlib.Path(__file__).parent / '../Library/Python/3.10/lib/python/site-packages/en_core_web_sm/en_core_web_sm-3.7.1'
# nlp = spacy.load(path)

# This worked ultimately!: After I dragged over the whole actual en_core folder into this venv and imported it
nlp = en_core_web_sm.load()
tokenizer = RegexpTokenizer(r"[^.?!]+")


def sent_tokenize(text):  # Split text into sentences and remove delimiters like ? or . or ! etc.
    return list(map(str.strip, tokenizer.tokenize(text)))


def clean_string(text, stem="None"):  # The text cleaning function

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':  # If user wishes to use stemming methodology, should be noted that stemming
                        # unlike lemmatizing might produce meaningless root forms of words
        stemmer = PorterStemmer() # using PorterStemmer to *reduce* invalid outcomes in data from
                                  # aggressive nature of stemming process
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':  # Lemmatizing determines word's Part of Speech before reducing to
                         # *meaningful* root form
        #
        # Note on using WordNet Lemmatizer: it assumes each word to be a noun by default.
        # Therefore, it requires an argument of the form (word, pos="r") for adverb etc.
        # to function maximally; or it needs multiple passes.
        #
        # pos_tag may be imported from nltk and used to automate fetching of Part of Speech (POS) tags.
        #
        lem = WordNetLemmatizer()  # created from nltk (imported at the beginning)
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]

    # When the user wants to be most thorough if slower, Spacy may be used
    # However, note: Spacy's max text length per-processing is 1*10^6 characters, to prevent memory allocation errors.
    # Therefore, to prevent hassle, I used lemmatizing.
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string


def create_dataframe(text, cleaning_method):  # Outputs a pandas dataframe from a text, with one column containing
                              # sentences, other column containing the cleaned sentence for analysis. For input,
                              # takes in the text and the NLP cleaning method to use: Stemming, Lem, or Spacy.

    list_of_sentences = sent_tokenize(text)  # contains the text's sentences in order
    cleaned_sentences = {sentence : clean_string(sentence, cleaning_method) for sentence in list_of_sentences}

    dataframe_columns = ["sentences", "cleaned sentences"]
    dataframe = pd.DataFrame.from_dict(cleaned_sentences, orient="index", columns=dataframe_columns)
    return dataframe

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# WORKS !!
# clean_text = clean_string("hello bro enekdho ,, i am going far", stem="Lem")
# print(clean_text)
# WORKS !!
# Note: for this example also, Spacy trumps Lem because it reduces going to go which is correct, unlike Lem.

# database_filepath = '/kaggle/input/scifi-stories-text-corpus/internet_archive_scifi_v3.txt'  # A holdover from Kaggle

# Use webscraping to retrieve the database into a temporary file?
database_filepath = '/Users/anuragtripathi/Downloads/internet_archive_scifi_v3.txt'
original_text = open(database_filepath, "r").read().lower()  # Text length: 94040093 characters

start_time = time.time()

# clean_text = clean_string(original_text, "Lem")  # Damn! Took plenty of time even for stemming! But succeeded!! Yay!

end_time = time.time()

elapsed_time = end_time - start_time

# print("Cleaning successful!\n\nTime taken for cleaning:", elapsed_time, "s")
# print("First 5k words of cleaned text:", clean_text[1:5000])

# Note: Could also make three separate empty files and write a cleaned text using one method per file,
# to the files, so don't need to process over and over again.

# Lemmatizing the words to be searched for
# Questions: how to divide text into sentences. how to search for three words appearing in one sentence.
# Now realized, I can use a pandas dataframe, dividing the text into sentences and each sentences into words.

