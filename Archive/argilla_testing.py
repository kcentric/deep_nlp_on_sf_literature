# Old Code from when I was trying to use Argilla for annotation

import pandas
import argilla as rg
from datasets import load_dataset
import pandas as pd
import spacy
from tqdm.auto import tqdm

custom_additions_to_stopwords = []
num_topics = 10
num_passes = 20

ner_df = pandas.read_csv("NER_output1.csv")
entity_list = ner_df['Entity'].values.astype(str).tolist()
print(len(entity_list))
with open("entity_list.txt", "w+") as txt:
    txt.write(str(entity_list))

print("Initializing Argilla")
# Replace api_url with the url to your HF Spaces URL if using Spaces
# Replace api_key if you configured a custom API key
# Replace workspace with the name of your workspace
# rg.init(api_url="http://localhost:6900", api_key="owner.apikey,", workspace='admin')

records = rg.read_datasets(
    load_dataset(path="Argilla_Dataset", data_files="NER_output2.csv", split="train"),
    task="TokenClassification",
)

dataset = load_dataset("/Users/anuragtripathi/PycharmProjects/KaggleDAProject1"
                       "/Argilla_Dataset", split="train", streaming=True)

# Let's have a look at the first 5 examples of the train set.
pd.DataFrame(dataset.take(5))

nlp = spacy.load("en_core_web_lg")

# Creating an empty record list to save all the records
records_list = []

# Iterate over the first 50 examples of the Gutenberg dataset
for record in tqdm(list(dataset.take(50))):
    # We only need the text of each instance
    # text = record  # earlier, text = record['tok_context'] but my dataset doesn't have that key.
    # print(record)  # Wanted to see what 'record' was actually containing: studying Argilla :-)
    text = record['Sentence']  # wow! apparently, it absolutely maps my csv file, each element of it.

    if text is None:  # Was getting spaCy errors about getting a NoneType value, so creatively tested
                      # why that occurs
        print("Skipping record with NoneType text:", record)
        continue

    # spaCy Doc creation
    doc = nlp(text)

    # Entity annotations
    entities = [(ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

    # Pre-tokenized input text
    tokens = [token.text for token in doc]

    # Argilla TokenClassificationRecord list
    records_list.append(
        rg.TokenClassificationRecord(
            text=text,
            tokens=tokens,
            prediction=entities,
            prediction_agent="en_core_web_lg",
        )
    )

rg.log(records=records, name="sci-fi_spacy_ner")
