from LLM.LLM_Interactor import GPTFeeder
import pandas as pd
from time import time

#TODO: Send multiple prompts per request. 5 prompts per request.
# OR, just send 40 sentences per para. :) Same effect, day done in 200 prompts.

def convert_data_to_dataframe(data):
    """Takes in 'data' as a list of lists where the each inner list represents a sentence. The inner
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



#TODO: for loop

# # Make a paragraph
# para = ". ".join(sentences[130:180])
#                                       # 130 - 140 has concepts and technology, yes it is catching!
#                                       # with 130 - 230 is it too big?
#
# answer = gpt.send_prompt(para, write_response_to_file=False)  # Don't write to json. We will need
#                                                               # a Pandas dataframe for RoBERTa training

if __name__ == "__main__":
    gpt = GPTFeeder()  # Create GPTFeeder object with default settings

    # Bring in our "representative sentences": 8000 sample sentences from the 114k named-entity-containing
    # sentences, from across the dataset.
    with open("../TF-IDF/representative_sents_for_GPT_labelling.txt", "r") as sentences_file:
        as_string = sentences_file.read()  # read file as one string
        sentences = as_string.splitlines()  # split the string into the separate sentences
        print("Check number of sentences:", len(sentences))
        print("First few:", sentences[:30])

    sents = sentences[141:152]  # 181 kro.    130-141, 141-152
    para = ". ".join(sents)
    print(para)
    start_t = time()
    answer = gpt.pipeline(para)
    # answers = [doc for doc in gpt.pipeline.pipe(sents, n_process=5)]

    # NOTE FROM OBSERVATION: IF I DO FOR LOOP THEN IT TAKES MY COMPUTER ON THE HOOK AND SO THE TIME IS A LOT
    # for sent in sents:
    #     answer = gpt.send_prompt(sent, write_response_to_file=False)
    #     answers.append(answer)

    # for i in range(len(sents) // 10):  # Iterate to send prompts in batches of 10 sentences only each time
    #     answer = gpt.send_prompt()

    # ents_with_labels = [(ent.text, ent.label_) for ent in answer.ents]
    #TODO: Hmm. The answer is coming from a paragraph, as the entities of a paragraph rather than sentence-wise.
    #TODO: Answer: lolus! I'm making a para and sending, lol. I could just send one by one, lolu!

    #TODO: Big todo. Divide the entities into SF and Non-SF only. Like this it is just catching Person always

    # dataframe = convert_data_to_dataframe(answer)
        # print(answer.text == para)
    # for answer in answers:
    print(answer.text)
    print(answer.ents)
    print([(ent.text, ent.label_) for ent in answer.ents])

    print(time() - start_t, "s")

