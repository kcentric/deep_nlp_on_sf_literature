# Taking the list of named entities from entity_list.txt, and the list of sentences of the whole text,
# to match named entities with their respective sentences and then eliminate duplicates so that we have
# as many sentences as named entities: a much shorter ultimate list than the original 3.5-million
# sentences. We should have ~135k sentences at the end.

# This sentence list can then be used much more effectively with an LLM to extract SF terms, than a mere
# list of named entities can. From a sentence list, we can join blocks of sentences into single paragraphs
# and feed them to an LLM, these blocks being arbitrarily long. We can then save the texts along with
# the LLM's results to a separate data-file.

import pandas as pd
import csv


def binary_search_word(sentence, sorted_word_list):
    """A binary search that takes in a sorted list of words or terms, along with a sentence, and sees
    if any of the words in the wordlist is present in the sentence.

    USUAL time complexity: O(log n) IF "sentence" is a small constant (<20 words for ex) long. As intended.
    Max POTENTIAL time complexity: O(n * log n) if sentence is a huge list itself."""
    # Split the sentence into words
    words_in_sentence = sentence.split()

    # Perform binary search for each word in the sentence
    for word in words_in_sentence:
        # Binary search implementation
        low, high = 0, len(sorted_word_list) - 1

        while low <= high:
            mid = (low + high) // 2
            mid_word = sorted_word_list[mid]

            if mid_word == word:
                return True  # Word found in the list
            elif mid_word < word:
                low = mid + 1
            else:
                high = mid - 1

    return False  # None of the words found in the list


def extract_list_of_sentences(csv_filepath, which_column, print_results=True):
    # Setting the filepath
    csv_file_path = csv_filepath

    # Specifying the column index we want to read (0-based index)
    column_index = which_column

    # Open the CSV file and read the specified column into a list
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        column_data = [row[column_index] for row in reader]

    # Display the list of strings
    if print_results:
        print(column_data[:50], "...")
        print("Num sentences:", len(column_data))

    return column_data


def find_sentences_with_entities(list_of_sentences, entity_list_filepath, print_results=True):
    """Note: Assumes that the text at given filepath is in the same format as the entity_list.txt
    bundled with the package"""
    with open(entity_list_filepath, 'r') as file:
        # Read the content of the file and remove leading/trailing whitespaces
        content = file.read().strip()

        # Remove single quotes and split the content into a list
        entities = [value.strip() for value in content.replace("'", "").split(",")]

    # Sort for binary search
    sorted_entity_list = sorted(entities)

    # Get a list containing all sentences that contain any of the named entities in them
    entity_sentences = [sent for sent in list_of_sentences if binary_search_word(sent, sorted_entity_list)]

    # Remove duplicates while preserving order
    seen = set()
    unique_entity_sentences = [sent for sent in entity_sentences if sent not in seen and not seen.add(sent)]

    if print_results:
        print(unique_entity_sentences[:50], "...")
        print("Num unique entity-containing sents:", len(unique_entity_sentences))

    return unique_entity_sentences

def are_csv_files_equal(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if DataFrames are equal
    return df1.equals(df2)

def checking_my_csv_files():
    file1_path = '../sci-fi_text_cleaned.csv'
    file2_path = '../sci-fi_text_cleaned1.csv'

    if are_csv_files_equal(file1_path, file2_path):
        print("The CSV files contain the same content.")
    else:
        print("The CSV files do not contain the same content.")


if __name__ == '__main__':
    # checking_my_csv_files()
    sentences = extract_list_of_sentences("../sci-fi_text_cleaned.csv", 1)  # extracting "Sentences" column
    uniques = find_sentences_with_entities(sentences, "entity_list.txt")

    with open("../LLM/sents_containing_named_entities.txt", "w+") as outfile:
        # Write each item from the list to a new line in the file
        for item in uniques:
            outfile.write("%s\n" % item)





