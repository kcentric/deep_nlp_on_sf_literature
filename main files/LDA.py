# NOTE: TODO: Say "word" instead of sentences. And refactor "document" to "documents"
# NOTE: TODO: Put quadgrams in output text file also?
# NOTE: TODO: Adjust saving so that it saves LDA model image to LDA folder.

"""Contains two classes:

1.) An LDA-based topic-model to extract "topics" or "discourses" out of the text (text being either one book or a corpus
of books in our case). Has a visualization method also.

2.) A MALLET-based LDA model that inherits from the original LDA (MALLET is known to have better accuracy).

Note: Uses multicore gensim model for (much) faster processing."""

PATH_TO_MALLET = '../mallet-2.0.8/bin/mallet'  # We will need this in case we want to make a MALLET-based LDA model
                                            # (separate class below)

import matplotlib
import numpy
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # important!
import matplotlib.pyplot as plt

# Enabling logging for gensim
import logging
import warnings  # Helpful to alert the user about some warnings in some cases

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global constants
NUM_TOPICS = 10  # Number of topics to train the LDA model on
NUM_PASSES = 50  # Set your optimal number of passes. Usually, increase number of passes as the size of corpus decreases


class LDA:

    """Take in a list of tokenized documents (list of lists), then generate bigrams, trigrams, and quadgrams,
    a word-id to word-frequency dictionary and a corresponding mapping for each sentence: proceed to create an LDA model
    by training it on this data."""

    def __init__(self, clean_text, use_bigrams=False, use_trigrams=False,
                 num_topics=NUM_TOPICS, num_passes=NUM_PASSES):
                                     # Extra parameters in case want to test with only bigrams/trigrams
                                     # Take in a LIST (of lists) of tokenized documents representing the corpus. (Tokens
                                     # are units of meaning. A first-degree tokenization of a text gives us a list of sentences,
                                     # and tokenizing each of those sentences gives us a list of words). Each document is a list
                                     # of words/tokens: so if the corpus is a list of books, each book is a (huge) list of words.
        self.document = clean_text
        self.bigram = gensim.models.Phrases(self.document, min_count=5, threshold=100)  # higher threshold fewer phrases.
        self.trigram = gensim.models.Phrases(self.bigram[self.document], threshold=100)
        self.quadgram = gensim.models.Phrases(self.trigram[self.document], threshold=100)
        self.num_topics = num_topics  # Number of topics to define document into
        self.num_passes = num_passes  # Number of passes to make per document for training

        # According to gensim.models.phrases.FrozenPhrases documentation, this will make things "much smaller and faster"
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)
        self.quadgram_mod = gensim.models.phrases.Phraser(self.quadgram)
        # Note: Personally I was interested in quadgrams only because of the nature of my text; however the model retains
        #       the capability to process using only bigrams/trigrams

        # Initialize "empty" vars that will contain our bigrams, trigrams, and/or quadgrams after processing
        # Each element of these lists will be a book/document represented as a list of quadgrams/trigrams/bigrams.
        # So we will have a list of lists.
        self.bigrams = []
        self.trigrams = []
        self.quadgrams = []

        self.lda_model = None  # Empty var to hold our trained LDA model later
        self.doc_lda = None

        # Perplexity and coherence scores to judge how good a given topic model is
        self.model_perplexity = 0.0
        self.coherence_score = 0.0

        # A variable to which we will later assign a dictionary containing words with their unique ids
        self.id_word_map = Dictionary()  # A Gensim corpora Dictionary, which is *not exactly a "dict"*

        self.term_document_frequency = []  # To hold a mapping of word-ids to word-frequencies as a list of tuples

        self.process(use_bigrams, use_trigrams)  # Make bigrams, trigrams and quadgrams (quadgrams are default,
                                                 # bigrams/trigrams must be specified)
        self.model()  # Create and assign the LDA model
        self.save_model_to_disk()

        # Find accuracy scores
        self.compute_coherence()
        self.compute_perplexity()

        # Save output and other info to file
        self.save_output_to_file()

    def make_bigrams(self):
        bigram_mod = self.bigram_mod
        texts = self.document
        return [bigram_mod[text] for text in texts]

    def make_trigrams(self):
        bigram_mod = self.bigram_mod
        trigram_mod = self.trigram_mod
        texts = self.document
        return [trigram_mod[bigram_mod[text]] for text in texts]

    def make_quadgrams(self):
        trigram_mod = self.trigram_mod
        quadgram_mod = self.quadgram_mod
        texts = self.document
        return [quadgram_mod[trigram_mod[text]] for text in texts]

    def process(self, generate_bigrams, generate_trigrams):
        if generate_bigrams is True and generate_trigrams is False:
            self.bigrams = self.make_bigrams()
            return
        elif generate_trigrams is True:  # If the user sets both bigram and trigram generation to True, then
                                         # we default to just generating trigrams for the model, assuming that
                                         # trigrams in general produce better results than bigrams alone.
            self.trigrams = self.make_trigrams()
        else:
            self.quadgrams = self.make_quadgrams()

    def model(self):
        """Creates the LDA model. Assumes that input is *already* lemmatized."""
        # texts = self.document  # This line was responsible for the model FAILING to produce topics rightly, because
                                 # here it was taking in the original doc instead of the quadgrams I'd so assiduously made!
                                 # Fixed below!

        # Has preferential treatment for quadgrams over trigrams over bigrams
        if len(self.quadgrams) > 0:
            texts = self.quadgrams
        elif len(self.trigrams) > 0:
            texts = self.trigrams
        else:
            texts = self.bigrams

        # Create dictionary, assigning unique word-id to each word
        self.id_word_map = corpora.Dictionary(texts)

        # Create a word-id to word-frequency mapping for each sentence. This will be used as input by the LDA model
        self.term_document_frequency = [self.id_word_map.doc2bow(text) for text in texts]  # Term Document Frequency

        # A human-readable format of the term-frequency map up to the third sentence
        printable = [[(self.id_word_map[word_id], freq) for word_id, freq in cp] for cp in self.term_document_frequency[:3]]
        # print(printable)  # For user

        # Note: Using the LDA Multicore model to process using parallelization on CPU.
        #       The "workers" parameter is set to 5 to optimize for Apple M1 Pro 6-core
        #       chip specifically. CHANGE this parameter based on what system is being used.
        self.lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=self.term_document_frequency,
                                                                 #workers=1,
                                                                 id2word=self.id_word_map,
                                                                 num_topics=self.num_topics,
                                                                 random_state=100,
                                                                 chunksize=100,
                                                                 passes=self.num_passes,
                                                                 per_word_topics=True)

        # self.doc_lda = self.lda_model[self.term_document_frequency]

    def save_model_to_disk(self):
        self.lda_model.save('lda.model')

    def save_output_to_file(self):
        with open('../Data Files (Readable)/LDA Files/lda_output.txt', 'w+') as output_file:
            topics = self.lda_model.top_topics(self.term_document_frequency)
            output_file.write("MODEL RESULTS AND MODEL DATA\n\nMODEL RESULTS:\n\nTopics:\n" +
                              '\n'.join('%s %s' % topic for topic in topics))
            output_file.write(f"\nModel perplexity: {self.model_perplexity}\nCoherence score: {self.coherence_score}")
            output_file.write(f"\n\nMODEL DATA:\nBigrams:\n{self.bigrams}\nTrigrams:\n{self.trigrams}"
                              f"\nQuadgrams:{self.quadgrams}")

    def compute_perplexity(self):
        self.model_perplexity = self.lda_model.log_perplexity(self.term_document_frequency)

    def compute_coherence(self):
        coherence_model = CoherenceModel(model=self.lda_model, texts=self.document, dictionary=self.id_word_map,
                                         coherence='c_v')
        self.coherence_score = coherence_model.get_coherence()

    # Print functions for user convenience
    def print_bigrams(self):
        print(f"First 100 Bigrams:\n{self.bigrams[:100]}")

    def print_trigrams(self):
        print(f"First 100 Trigrams:\n{self.trigrams[:100]}")

    def print_quadgrams(self):
        print(f"First 100 Quadgrams:\n{self.quadgrams[:100]}")

    def print_topics(self):
        """Output our model by printing the keywords for each topic with their weightage to the topic."""
        lda_model = self.lda_model
        print("\nTopics are:\n")
        pprint(lda_model.print_topics())

    def print_scores(self):
        print("\nModel Perplexity:", self.model_perplexity)
        print("\nCoherence score:", self.coherence_score)

    def visualize_model(self):
        """Generate a visualization of the LDA model using pyLDAvis."""
        # pyLDAvis.enable_notebook()  # iPython-designed line; don't need it unless using a notebook
        vis = pyLDAvis.gensim.prepare(self.lda_model, self.term_document_frequency, self.id_word_map, mds='tsne')  # add mds specification to prevent TypeError
        pyLDAvis.show(vis, local=False)  # Will only work with iPython
        # html = pyLDAvis.prepared_data_to_html(vis)
        # pyLDAvis.save_html(html, "LDA_visualization.html")

    # A clever version of __str__ that tries to work around my mistake of putting prints in it rather than a return :-)
    def __str__(self):
        self.print_bigrams()
        self.print_trigrams()
        self.print_quadgrams()
        self.print_topics()
        self.print_scores()
        return ""







