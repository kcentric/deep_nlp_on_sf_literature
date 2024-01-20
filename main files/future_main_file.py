# MAJOR NOTE: THIS IS *NOT* THE ACTUAL MAIN FILE RIGHT NOW! MORE OF A TESTING FILE. BUT THIS *WILL*
#  BE MADE INTO THE MAIN FILE BEFORE FINAL COMMIT, AS INDICATED BY THE FILE'S NAME.

# TODO: MAKE FOLDERS CONTAINING EACH MODULE AND RELEVANT TXT FILES ALONG WITH IT.
# TODO: FOR EG, MAKE LLM_INTERACTOR FOLDER, CONTAINING LLM_INTERACTOR.PY AND SENTS_CONTAINING_NAMED_ENTITIES.TXT
# TODO: WHY? BCZ THEN I CAN MAKE A SEPARATE FILE WITH THE JSON STRING AND CONFIG STRING THAT'S IN THE
# TODO: LLM FOLDER, SO EVERYTHING IS MODULARIZED.
# TODO: MAYBE MAKE SCRIPTS AND PUT FUNCTION FILES IN THEM?

# TODO: Keep Project files of Argilla but delete the anuragtripathi folder ones. Delete all Doccano
# TODO: Pass in custom additions to stopwords also into LDATrainer and from there to CorpusProcessor;
# TODO: if not passed in, custom_stopwords is empty.
# TODO: or LDATrainer by default says working on stories, and then we have those stopwords and use NER also.
# TODO: say use_ner_in_cleaning parameter is false by default but can use.

import os
import spacy
# from doccano_preparation import csv_to_json
#
# csv_to_json(csv_file_path='NER_output1.csv',
#             json_file_path='JSON_Files_for_Doccano/NER_output_for_Doccano.json',
#             split_into_5=True, sample_size=100)

# os.environ["OPENAI_API_KEY"] = api_key
# print(os.environ.get("OPENAI_API_KEY"))
# nlp = spacy.load("en_core_web_lg")
# llm = nlp.add_pipe("llm_textcat")
# llm.add_label("INSULT")
# llm.add_label("COMPLIMENT")
# doc = nlp("You look gorgeous!")
# print(doc.cats)


# Example list of sentences
list_of_sentences = ["Bert ate chocolate", "Alice met brother", "Cat ate dog"]

# Create a list of tuples with original sentences and their index
sentences_with_index = list(enumerate(list_of_sentences))

# Sort the list of tuples based on the sentences
sorted_sentences_with_index = sorted(sentences_with_index, key=lambda x: x[1])  # x[1] is the second element
# of each tuple, which is the
# sentence

# Now get a list of the sorted sentences themselves (list of strings)
sorted_sentences = [sent for _, sent in sorted_sentences_with_index]  # Clever use of _ to not use the index in the
# tuple at all. _, sent represents each tuple

# Print the result
print("Sorted list of sentences:", sorted_sentences[:50], "...")

# # List of sentences containing any entities, in sorted order. List also might contain duplicates
# entity_sentences_sorted = [sent for sent in sorted_sentences for value in values_list if binary_search(sent, value)]
#
# # Remove duplicates while preserving order
# seen = set()
# unique_entity_sentences = [sent for sent in entity_sentences if sent not in seen and not seen.add(sent)]
