# Notes on optimization: I initially used the "en_core_web_sm" model for Spacy nlp,
#                        which enabled me to perform NER on my full corpus in a record
#                        1185 seconds. Later, I decided to switch to "en_core_web_lg"
#                        which processed in 1507 seconds - still a great log n complexity.

import spacy
import pandas
import time

# Path to file where we can store the NER output
FILEPATH = 'NER_output.csv'
FILEPATH2 = 'NER_output1.csv'
FILEPATH3 = 'NER_output2.csv'

## UNCOMMENT to download spacy pipeline BEFORE running file for the first time.
# spacy.cli.download("en_core_web_lg")  # Note: Downloaded "lg" or "large" model for
                                        #       best performance. Download and load
                                        #       "en_core_web_sm" to use less disk space.
# Load spacy
nlp = spacy.load("en_core_web_lg")


class NER:
    def __init__(self, text, text_already_cleaned=True, parallelize_training=True, num_workers=-1):

        """Default time complexity: O(log n). Takes in a "text" as input. Assumes that *text* is a list of cleaned
        strings (punctuation removed, lemmatized, etc.) with each string representing a sentence. Extracts Named
        Entities and creates a dataframe.

        Runtime is optimized (by default) by using parallelization, which can be adjusted (*num_workers* parameter) for
        different numbers of CPU cores. To turn off this feature, set *parallelize_training* to False.

        IMPORTANT: If *text_already_parsed* is set to False, will assume that *text* is a STRING and proceed
        to perform nlp processing (Spacy model) on it, which can significantly increase processing time. Thus,
        it is recommended to input the expected value types."""

        self.cleaned_sentences = []
        self.dataframe = None
        if text_already_cleaned:
            self.cleaned_sentences = text
            if parallelize_training is True:
                self.run_ner_with_multithreading(num_workers)
            else:
                self.run_ner()

    def run_ner_with_multithreading(self, num_workers):
        entities = []
        type_entity = []
        sentences = []

        # Use a spacy pipe method for parallel processing to produce a list of doc objects
        parsed_sentences = list(nlp.pipe(self.cleaned_sentences, n_process=num_workers))

        # Extract entities from each spacy doc object, perform checks and appends
        for sent in parsed_sentences:
            for ent in sent.ents:
                if ent.text not in entities:
                    entities.append(ent.text)
                    sentences.append(sent)
                    type_entity.append(ent.label_)
        self.dataframe = pandas.DataFrame({'Sentence': sentences, 'Entity': entities,
                                           'Entity_type': type_entity})
        print('The total number of entities detected were:{}'.format(len(self.dataframe)))
        print(self.dataframe)

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
    from pathlib import Path

    csv_file_path = "../sci-fi_text_cleaned1.csv"
    if Path(csv_file_path).is_file() is False:  # Generate the required csv.
        from CorpusProcessor_with_NER import CorpusProcessor
        textprocessor = CorpusProcessor()
        textprocessor.create_dataframe(write_to_csv=False)  # Peculiarity of create_dataframe method.
                                                            # If write_to_csv is True, it writes
                                                            # to the default "sci-fi_text_cleaned.csv" file
                                                            # only.
        # Write to our intended path
        textprocessor.write_dataframe_to_csv(csv_file_path)

    # Timekeeping
    start_time = time.time()

    # Bring in the sentences we'll use for NER
    df = pandas.read_csv(csv_file_path)
    cleaned_sentences = df['Cleaned sentences'].astype(str).tolist()

    print("Please verify.\nNumber of sentences:", len(cleaned_sentences))
    print("First 50 sentences:", cleaned_sentences[:50])

    print("Please wait. Performing NER.")
    ner_obj = NER(cleaned_sentences)
    ner_obj.write_to_csv(FILEPATH3)

    print("Time taken:", time.time() - start_time, "seconds")
