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

# Lines that didn't work:

# nlp = spacy.load('en_core_web_sm') # Note: Install using "python -m spacy download en_core_web_sm" command on terminal
# nlp = spacy.load('./Library/Python/3.10/lib/python/site-packages/en_core_web_sm/en_core_web_sm-3.7.1')

# path = pathlib.Path(__file__).parent / '../Library/Python/3.10/lib/python/site-packages/en_core_web_sm/en_core_web_sm-3.7.1'
# nlp = spacy.load(path)

# This worked ultimately!: After I dragged over the whole actual en_core folder into this venv and imported it
nlp = en_core_web_sm.load()
tokenizer = RegexpTokenizer(r"[^.?!]+")


def sent_tokenize(text):  # Split text into sentences and remove delimiters like ? or . or ! etc.
    # return list(map(str.strip, tokenizer.tokenize(text)))
    return list(map(str.lower, tokenizer.tokenize(text)))


def clean_string(text, only_remove_line_breaks=False, stem="None"):  # The text cleaning function

    # final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)
    if only_remove_line_breaks:  # allows the user to just return the text with line breaks removed
        return text

    # Remove punctuation
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

# print(clean_string("Hello. My name is Krishna; I am the maker of this project."
#                    "vvv the a an,", only_remove_line_breaks=True))


def create_dataframe(text, cleaning_method):  # Outputs a pandas dataframe from a text, with one column containing
                              # sentences, other column containing the cleaned sentence for analysis. For input,
                              # takes in the text and the NLP cleaning method to use: Stemming, Lem, or Spacy.

    list_of_sentences = sent_tokenize(clean_string(text, only_remove_line_breaks=True))  # contains the text's sentences in order
    cleaned_sentences = [clean_string(sentence, stem=cleaning_method) for sentence in list_of_sentences]

    data_dict = {
      "Sentences": list_of_sentences,
      "Cleaned sentences": cleaned_sentences,
    }
    dataframe = pd.DataFrame(data_dict)
    return dataframe


database_filepath = '/Users/anuragtripathi/Downloads/internet_archive_scifi_v3.txt'
original_text = open(database_filepath, "r").read().lower()  # Text length: 94040093 characters

# Just to measure time
start_time = time.time()

# Clean the text by splitting it into sentences and creating a dataframe with the sentences and their cleaned versions
clean_text = create_dataframe(original_text, "Lem")  # Damn! Took plenty of time even for stemming! But succeeded!! Yay!

# Write the dataframe to a csv to that we don't need to take time for cleaning again
new_file = open('../../../sci-fi_text_cleaned_v1.csv', "w+")
data_as_csv = clean_text.to_csv('sci-fi_text_cleaned_v1.csv')

end_time = time.time()

elapsed_time = end_time - start_time

print("Time taken for cleaning and organizing:", elapsed_time, "s")
print(f"Pandas Dataframe:\n{clean_text}")

