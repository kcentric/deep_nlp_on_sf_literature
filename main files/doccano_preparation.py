import csv
import json
import random

def csv_to_json(csv_file_path, json_file_path, split_into_5=True, sample_size=100):

    """
    Convert data from a CSV file to JSON format, optionally splitting it into five portions.

    Params:
     - csv_file_path (str): Path to the input CSV file.
     - json_file_path (str): Path to the output JSON file.
     - split_into_5 (bool, optional): If True, the data is split into five portions and saved in separate JSON files.
     - sample_size (int or None, optional): Number of sentences to randomly sample from each portion. If set to None,
     the entire portion is included in the JSON file. This argument is only necessary if *split_into_5* is True.

    Example:
    csv_to_json('input_data.csv', 'folder/output_data.json', split_into_5=True, sample_size=None)
    csv_to_json('input_data.csv', 'folder/output.json', split_into_5=True, sample_size=200)
    """

    # Read CSV file and convert it to a list of dictionaries
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Use this line if '':'0' doesn't come out is the first key-value pair of the first row
                                                       # of data. (Check using print statement)
        # data = [row for row in csv_reader]
        # print(data[0])

        # Eliminate non-useful first key-value pair ('':'0', '':'1' etc.) from each dictionary
        data = [dict(list(row.items())[1:]) for row in csv_reader]
        # print("First 4 elements of data:")
        # print(data[:4])
        if split_into_5 is False:
            data = [
                    {
                        "Sentence": row["Sentence"],
                        "['Entity', 'Entity_type']": [row["Entity"], row["Entity_type"]],
                    }
                    for row in data
                ]
            # Write the list of dictionaries containing this portion to the JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=2)

        else:

            # Split data into 5 equal portions and write to separate json files
            for i in range(1, 6):

                # Create a portion beginning from where the previous ended
                start_pos = 27054 * (i - 1)
                end_pos = 27054 * i
                portion = data[start_pos:end_pos]

                # Include the last few sentences (after 135,270th sentence) in the fifth portion also
                if end_pos == 135270:
                    portion += data[end_pos:]

                # Now adjust the portion for Doccano's format (allows only two columns)
                portion = [
                    {
                        "Sentence": row["Sentence"],
                        "['Entity', 'Entity_type']": [row["Entity"], row["Entity_type"]],
                    }
                    for row in portion
                ]

                if sample_size is not None:
                    # Select 100 random sentences from the portion
                    portion = random.sample(portion, 100)

                #print(portion[:4])

                # Remove "json" from the given json filepath name to get a stub for the filename
                if json_file_path[-4:] == "json":  # Just to make sure if user hasn't given a "/" at the end
                    json_filename_stub = json_file_path[:-4]
                else:
                    print("Please input the json file path without a slash at the end. Input must be"
                          "something like: '/directory/jsonfile.json' rather than '/directory/jsonfile.json/'")
                    return

                # Generate a good filename
                filename = json_filename_stub + f"_Portion_{i}.json"

                # Write the list of dictionaries containing this portion to the JSON file
                with open(filename, 'w') as json_file:
                    json.dump(portion, json_file, indent=2)

if __name__ == '__main__':
    csv_to_json('NER_output1.csv', 'JSON_Files_for_Doccano/NER_output_for_Doccano.json',
                split_into_5=True)
