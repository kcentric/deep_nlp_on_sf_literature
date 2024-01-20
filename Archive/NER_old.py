import spacy
from concurrent.futures import ThreadPoolExecutor
#from CorpusProcessor_with_NER import CorpusProcessor
import random
import matplotlib.pyplot as plt
#TODO: Complete this class? Or leave, just make it simpler.
#TODO: pandas is only temporarily here. shift that functionality to corpusprocessor
#TODO: adjust all file creations to go to data files folder?
import pandas
import time

# Path to the file where we can store the NER output
FILEPATH = '../NER_output.csv'
FILEPATH2 = 'NER_output1.csv'

# spacy.cli.download("en_core_web_lg")  # The "large" (lg meaning large) model. Uncomment
                                        # to load for more thorough model, if desired.
                                        # Large model's predictions are usually only incrementally
                                        # better than "sm" (small) model, however, despite taking up
                                        # significantly more disk space.

## UNCOMMENT to download spacy pipeline BEFORE running file for the first time.
# spacy.cli.download("en_core_web_sm")

# Load spacy
nlp = spacy.load("en_core_web_sm")


class NER:
    def __init__(self, text, text_already_cleaned=True, multithreading=True):

        """Takes in a "text" as input. Assumes that *text* is a list of cleaned strings (punctuation removed,
        lemmatized, etc.) with each string representing a sentence. Extracts Named Entities and creates a dataframe.

        If *multithreading* is True, optimizes runtime by using ThreadPoolExecutor, which can be adjusted for
        different numbers of CPU cores.

        IMPORTANT: If *text_already_parsed* is set to False, will assume that *text* is a STRING and proceed
        to perform nlp processing (Spacy model) on it, which can DRASTICALLY increase processing time. Thus,
        it is recommended to input the expected value types."""

        self.cleaned_sentences = []
        self.dataframe = None
        if text_already_cleaned:
            self.cleaned_sentences = text
            if multithreading:
                self.run_ner_with_multithreading()
            else:
                self.run_ner()
            # with ThreadPoolExecutor(max_workers=5) as executor:
            #     processed_docs = list(executor.map(self.run_ner))
            # #self.run_ner()

    def run_ner_with_multithreading(self):
        self.entities = []
        self.type_entity = []
        self.sentences = []

        with ThreadPoolExecutor as executor:
            # noinspection PyTypeChecker
            parsed_sentences = list(executor.map(self.operate_on_sentence, self.cleaned_sentences))

        self.dataframe = pandas.DataFrame({'Sentence': self.sentences, 'Entity': self.entities,
                                           'Entity_type': self.type_entity})
        print('The total number of entities detected were:{}'.format(len(self.dataframe)))
        print(self.dataframe)

    def operate_on_sentence(self, sentence):
        parsed_sentence = nlp(sentence)
        for ent in parsed_sentence.ents:
            if ent.text not in self.entities:
                self.entities.append(ent.text)
                self.sentences.append(sentence)
                self.type_entity.append(ent.label_)
        return parsed_sentence

    def run_ner(self):
        entities = []
        type_entity = []
        sentences = []

        for sent in self.cleaned_sentences:
            parsed_sentence = nlp(sent)  # Get sentence as Spacy doc object

            # Extract entities from the doc object, perform checks and appends
            for ent in parsed_sentence.ents:
                if ent.text not in entities:
                    entities.append(ent.text)
                    sentences.append(sent)
                    type_entity.append(ent.label_)

        self.dataframe = pandas.DataFrame({'Sentence': sentences, 'Entity': entities, 'Entity_type': type_entity})
        print('The total number of entities detected were:{}'.format(len(self.dataframe)))
        print(self.dataframe)

    def write_to_csv(self, filepath):
        self.dataframe.to_csv(filepath)


if __name__ == '__main__':
    csv_file_path = "../sci-fi_text_cleaned.csv"  # This file MUST have been created using CorpusProcessor before this

    # Timekeeping
    start_time = time.time()

    # Bring in the sentences we'll use for NER
    df = pandas.read_csv(csv_file_path)
    cleaned_sentences = df['Cleaned sentences'].astype(str).tolist()

    print("Please verify.\nNumber of sentences:", len(cleaned_sentences))
    print("First 50 sentences:", cleaned_sentences[:50])

    print("Please wait. Performing NER.")
    ner_obj = NER(cleaned_sentences[:50000])
    ner_obj.write_to_csv(FILEPATH2)

    print("Time taken:", time.time() - start_time, "seconds")
