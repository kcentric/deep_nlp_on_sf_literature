from LLM_Interactor import GPTFeeder
from transformers import RobertaTokenizer
import re
from json import JSONDecodeError
import pandas as pd
import json
from time import time

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def tokenize_sents_or_strings(sentences):
    """Tokenize a list of sentences/strings using RoBERTa's tokenizer and return a list of lists"""
    tokenized_sentences = []
    global tokenizer
    for sent in sentences:
        tokenized_sentences.append(tokenizer.tokenize(sent))
    return tokenized_sentences


def generate_raw_training_data(write_to_file=True, filename="GPTResponses.json"):
    """Extract the list of 8000 sentences from a prepared file, perform SF Entity recognition on them
    using GPT-3.5, and store the responses in a well-structured json file. This can then be used as
    training data to fine-tune RoBERTa or another similar model.

    NOTE: When actually creating training data, *write_to_file* should always be left to True. For testing,
    it's good to set it to false so that previously stored data is not messed up."""

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

    num_iterations = len(sentences) // 10  # We'll batch the sentences into paragraphs of 10 sentences
                                          #  to reduce number of API calls
    # print("num_iterations", num_iterations)
    for i in range(num_iterations):  # If we have 8000 sentences for example, we will send 800 "paragraphs" of 10
                                     #  sentences each to the LLM, for a total of 8000

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

        if write_to_file is True:
            # Dump each time so that if we run against an API usage limit we still will have everything in file
            with open(filename, "w+") as outfile:
                json.dump(answers, outfile, indent=2)


def convert_to_roberta_dataframe(data):
    """Takes in 'data' as a list of lists where each inner list represents a sentence. The inner
    list is further made up of two-element lists (which could be replaced with tuples) where each
    tuple/list consists of a word with its NER label. For example:
    data = [[['He', 'O'], ['likes', 'O'], ['Paris', 'B-LOC']], [['She', 'O'], ['enjoys', 'O'], ['soccer', 'O']]].

    Returns a pandas dataframe where each row is a sentence_id with a word from the sentence it represents,
    and the word's label. Suitable for training with the *simpletransformers* library."""
    # Uncomment below line for a test value of data and run the function with any value as an argument, to see
    # how the dataframe is supposed to look like
    # data = [[['He', 'O'], ['likes', 'O'], ['Paris', 'B-LOC']], [['She', 'O'], ['enjoys', 'O'], ['soccer', 'O']]]

    rows = []
    for sentence_id, sentence in enumerate(data):
        for word_label in sentence:
            word, label = word_label  # unpack the list into word and label
            rows.append((sentence_id, word, label))

    df = pd.DataFrame(rows, columns=["sentence_id", "words", "labels"])
    return df


def bring_in_data_as_list_of_dicts(json_filepath, all_sentences):
    """Fetch data from a JSON file formatted according to the specifications of the *create_training_data* function.
    The output should be a list of dictionaries, where each dictionary consists of a sentence paired with a list
    containing its corresponding entities and labels. The format of the list should be [(Entity, Label),
    (Entity, Label), (Entity, Label)]."""
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

        # Print statement for debugging purposes
        # print("Paragraph span:", [start_index, end_index])

        # Sentences from the original list that compose this paragraph
        sentences = all_sentences[start_index:end_index]

        entities = dictionary["entities_in_para"]
        sentence_entity_map = {}
        # Now that we have the sentences the entities were extracted from, let's assign the entities
        for sentence in sentences:
            its_entities = []
            for entity in entities:
                if entity[0] in sentence:
                    its_entities.append((entity[0], entity[1]))  # A tuple containing the entity with its label
            sentence_entity_map[sentence] = its_entities

        list_of_dicts.append(sentence_entity_map)

    return list_of_dicts


def list_of_dicts_to_list_of_lists(dict_list, label_the_unlabeled=True):
    """Take in a list of dictionaries containing key-value pairs of the format *"sentence": [(Entity, Label),
    (Entity, Label)]*. If *label_the_unlabeled* is True, explicitly label all unlabeled tokens as "==NONE==". Return a list
    of lists of the format [ [("Token", "Label"), ("Token, "Label"), ("Token", "Label")], [("Token, "Label")] ] Each
    inner list represents a sentence which has been tokenized and labeled. Special character 'Ġ' will represent spaces
    where a new token begins, since we use RobertaTokenizer.

    Note: If *label_the_unlabeled* is False, will only output already-labeled tokens. This means that "sentences" might
    be broken up. Recommendation is usually to keep *label_the_unlabeled* True."""

    output_list = []  # We will return this

    # One dictionary to contain all our sentences. We will merge the input dict_list here for convenience
    super_dict = {}
    for dictionary in dict_list:
        super_dict.update(dictionary)
    # print("Super dict length:", len(super_dict))

    global tokenizer  # The Roberta Tokenizer
    for key in super_dict:  # Each "key" is a sentence
        # print(type(key), key)
        tokenized_sentence = tokenizer.tokenize(key)
        original_entities = super_dict[key]

        # Tokenize the entities using RobertaTokenizer also, since 'entities' frequently consist of multiple words/tokens
        tokenized_entities = []
        for entity_tuple in original_entities:
            tokenized_entity = tokenizer.tokenize(entity_tuple[0])  # The first element of the tuple is the entity
                                                                    #  itself
            # Create a tuple containing the tokenized entity (a list), and the entity's label as its elements
            tokenized_entities.append((tokenized_entity, entity_tuple[1]))  # The second element of the entity
                                                                            # tuple is the entitiy's label
        ## Now 'tokenized_entities' should look like [ ( ["Token", "Token", "Token"], "Label" ) ]

        if label_the_unlabeled:
            temp_list = []  # We'll store here the fully-labeled sentence as a list of tuples
            for token in tokenized_sentence:
                # Initialize to_append with default values for the case where the token is not found in any entity
                label = "==NONE=="
                to_append = (token, label)
                for ent_tuple in tokenized_entities:
                    tokens_of_entity = ent_tuple[0]  # The list containing the tokens of the named entity
                    # Use the [1:] check for the (frequent) case where we might have a 'Ġ' inserted
                    #  by RobertaTokenizer in the sentence-token which doesn't exist in the entity-token
                    if any(t == token or t == token[1:] for t in tokens_of_entity):  # If for any t, t is in token, get True
                        # Update to_append with the label if the token is found in the entity
                        to_append = (token, ent_tuple[1])
                        break  # Break out of the loop if the token is found in any entity

                # Append a tuple to the 'temp_list'
                temp_list.append(to_append)
                ## Now, 'temp_list' should look like [("Token", "Label"), ("Token", "Label")]

        if label_the_unlabeled:  # Performing a separate check, just for cleanliness of code
            new_sentence_representation = temp_list
        else:
            # As warned in the documentation, we'll only give the tokenized entities as a representation of the sentence.
            # Break up tokenized_entities into the form [ ("Token", "Label"), ("Token, Label"), ("Token", "Label") ]
            #  where each token that shares a label is assigned separately.
            new_sentence_representation = []
            for ent_tuple in tokenized_entities:
                for string in ent_tuple[0]:
                    new_sentence_representation.append((string, ent_tuple[1]))

        output_list.append(new_sentence_representation)

    return output_list


if __name__ == "__main__":
    # Ask user if data for RoBERTa training is not already ready
    data_not_ready = input("Do you want to create the data first? Y or any other key: ")
    if data_not_ready == "Y":
        generate_raw_training_data()  # Run the function with default settings

    with open("../TF-IDF/representative_sents_for_GPT_labelling.txt", "r") as sentences_file:
        as_str = sentences_file.read()  # read file as one string
        all_sents = as_str.splitlines()  # split the string into the separate sentences

    list_of_dicts = bring_in_data_as_list_of_dicts("GPTResponses.json", all_sents)
    ready_for_dataframing = list_of_dicts_to_list_of_lists(list_of_dicts)
    df_for_roberta = convert_to_roberta_dataframe(ready_for_dataframing)

    print(df_for_roberta)  # TODO: See, with "Desert Cat" the issue with "G" is still coming.
                           #  But, FIRST check if RoBERTa is getting trained now. Solve that, then this


    # NOTE: Referring to list_of_dicts_to_list_of_lists function. I was wondering if there could be a
    #        possibility where an entity-tokenization has a 'Ġ' while its corresponding
    #        sentence-token doesn't  have a 'Ġ'. If you have the same doubt, this can
    #        be tested by running the 'list_of_dicts_to_list_of_lists' function and then
    #        seeing if the total number of labeled entities (treat sequences of tokens with the same
    #        label as one entity) in the output is equal to the total number of labeled entities which
    #        were there in the input.
