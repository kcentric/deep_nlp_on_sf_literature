#TODO (general): Remember to try turn all CorpusProcessor to log n time complexity also!

#TODO: Note on output; KMeans Clustering before NER round was DAMN effective; all useless things
# apparently got eliminated.

#TODO: Need to add first 200 and last 50 sentences to output.
# TODO: See if I can adjust this so that there's batch processing for 4-10 paragraphs which cuts time further

#TODO: Remember to create separate dev_data dataset for Roberta also
from LLM_Interactor import GPTFeeder
from transformers import RobertaTokenizer
import re
from json import JSONDecodeError
import pandas as pd
import json
from time import time

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_sents(sentences):
    """Tokenize a list of sentences using RoBERTa's tokenizer and return a list of lists"""
    tokenized_sentences = []
    global tokenizer
    for sent in sentences:
        tokenized_sentences.append(tokenizer.tokenize(sent))
    return tokenized_sentences

def create_training_data():
    """Extract the list of 8000 sentences from a prepared file, perform SF Entity recognition on them
    using GPT-3.5, and store the responses in a well-structured json file. This can then be used as
    training data to fine-tune RoBERTa or another similar model."""
    gpt = GPTFeeder()  # Create GPTFeeder object with default settings

    # Bring in our "representative sentences": 8000 sample sentences from the 114k named-entity-containing
    # sentences, from across the dataset.
    with open("../TF-IDF/representative_sents_for_GPT_labelling.txt", "r") as sentences_file:
        as_string = sentences_file.read()  # read file as one string
        sentences = as_string.splitlines()  # split the string into the separate sentences
        print("Check number of sentences:", len(sentences))
        print("First few:", sentences[:30])

    try:
        # Try to read the JSON file and load its content as a list of dictionaries
        with open("GPTResponses.json", "r") as old_responses:
            answers = json.load(old_responses)
    except (FileNotFoundError, JSONDecodeError):
        # Handle the case where the file doesn't exist, is empty or not valid JSON
        answers = []

    start_time = time()  # We'll measure time and keep updating the user throughout the progress

    for i in range(800):  # We will send 800 "paragraphs" of 10 sentences each to the LLM, for a total of 8000

        start_index = i*10  # Begin from the 0th index
        end_index = start_index + 10
        sents = sentences[start_index:end_index]
        para = ". ".join(sents)  # Create paragraph for easy GPT comprehension
        answer = gpt.pipeline(para)  # Get prompt response
        # Store answer info in a dictionary
        as_dict = {
            "span_of_para_in_sentence_list": [f"Beginning:{start_index}", f"End:{end_index}"],
            "entities_in_para": [(ent.text, ent.label_) for ent in answer.ents]
        }

        # Print good info so user can see satisfactory progress
        print("Result:")
        print(as_dict)
        answers.append(as_dict)

        print("Current time spent:", time() - start_time)

        # Dump each time so that if we run against an API usage limit we still will have everything in file
        with open("GPTResponses.json", "w+") as outfile:
            json.dump(answers, outfile, indent=2)

def convert_data_to_dataframe(data):
    """Takes in 'data' as a list of lists where each inner list represents a sentence. The inner
    list is further made up of two-element lists (which could be replaced with tuples) where each
    tuple/list consists of a word with its NER label. For example:
    data = [[['He', 'O'], ['likes', 'O'], ['Paris', 'B-LOC']], [['She', 'O'], ['enjoys', 'O'], ['soccer', 'O']]].

    Returns a pandas dataframe where each row is a sentence_id with a word from the sentence it represents,
    and the word's label. Suitable for training with the *simpletransformers* library."""
    # Uncomment below line for a test value of data and run the function with any value as an argument, to see
    # how the dataframe is supposed to look like
    data = [[['He', 'O'], ['likes', 'O'], ['Paris', 'B-LOC']], [['She', 'O'], ['enjoys', 'O'], ['soccer', 'O']]]

    rows = []
    for sentence_id, sentence in enumerate(data):
        for word_label in sentence:
            word, label = word_label  # unpack the list into word and label
            rows.append((sentence_id, word, label))

    df = pd.DataFrame(rows, columns=["sentence_id", "words", "labels"])
    return df

# TODO: Adjust so that it brings in data but also adjusts it all based on sentences rather than paras
# TODO: Change so that it takes Roberta-tokenized sentences in for easier entity matching
def bring_in_data(json_filepath, all_sentences):
    """Bring in training data created by GPT 3.5, stored in the same format as the *create_training_data*
    function stores its output. Returns a list of the format required by the *convert_data_to_dataframe*
    function."""
    with open(json_filepath, "r") as infile:
        dicts = json.load(infile)

    list_of_dicts = []
    for dictionary in dicts:
        # Extract the start and end indices of the sentences which compose the para
        indices = dictionary["span_of_para_in_sentence_list"]

        # Using regex to extract the number from the string containing the start index
        start_index = int((re.search(r'\d+', indices[0])).group())

        # Same for end index
        end_index = int((re.search(r'\d+', indices[1])).group())

        print("Paragraph span:", [start_index, end_index])

        # Sentences from the original list that compose this paragraph
        sentences = all_sentences[start_index:end_index]

        entities = dictionary["entities_in_para"]
        sentence_entity_map = {}
        # Now that we have the sentences the entities were extracted from, let's assign the entities
        for sentence in sentences:
            its_entities = []
            for entity in entities:
                if entity[0] in sentence:
                    its_entities += entity
            sentence_entity_map[sentence] = its_entities

        list_of_dicts.append(sentence_entity_map)

    # Merge all the dictionaries into one dictionary
    super_dict = {}
    for dict in list_of_dicts:
        super_dict.update(dict)
    print("Super_dict length:", len(super_dict))

    for key in super_dict:  # Iterate over each sentence
        sentence = key
        its_labels = super_dict[key]  # its_labels will refer to the list containing the sent's labels
        for word in sentence:
            continue
            #if super_dict[key]  #TODO: I have to adjust so that the labels are all in ORDER. How?

    #TODO: Check if the sents are in order. Merge the dictionaries into one. Remember that this order
    # is not the order in "all sentences". Then for all words which are not entities, mark them none
    # specifically, and split each sent into words with one id per sent. Or use RoBERTa tokenizer. Check

    return list_of_dicts

if __name__ == "__main__":
    # Ask user if data for RoBERTa training is not already ready
    data_not_ready = input("Do you want to create the data first? Y or any other key: ")
    if data_not_ready == "Y":
        create_training_data()

    with open("../TF-IDF/representative_sents_for_GPT_labelling.txt", "r") as sentences_file:
        as_str = sentences_file.read()  # read file as one string
        all_sents = as_str.splitlines()  # split the string into the separate sentences

    list_of_dicts = bring_in_data("GPTResponses.json", all_sents)
    for element in list_of_dicts:
        #continue
        print(element)

    #TODO: Now, I have Roberta-tokenized sentences, so I can match their indices with the untokenized
    # ones to match their entities (just in case some words in the entity-terms are removed)

    #TODO:Oh! Or, tokenize the entities also. Then if the token in the sent has a label, give it,
    # else label it None

    print(tokenize_sents(all_sents)[:30])
    # print(convert_data_to_dataframe(""))
