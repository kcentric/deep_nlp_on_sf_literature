# NOTE: Could also be extracting concepts like "distant past", "distant future" from the corpus/modeling

from api_key import key
from config_strings_2 import json_file, config_string

from typing import Iterable, List
import json
import openai
import pandas as pd
import spacy  # NOTE: Install spacy-llm FIRST. Documentation: https://github.com/explosion/spacy-llm
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.util import split_labels
import os
import json
import configparser
from spacy_llm.util import assemble
from datasets import load_dataset

# TODO: Adjust to incorporate "dont train" parameter
# TODO: create a separate class where "data preprocessed" and "chunksize" things can be used..?
# TODO (general): see if keeping capitalization in OG data helps?

def find_index_of_a_sentence(sent_list, word_string):
    """Helpful function enabling you to find the first sentence containing a particular group of words,
    in a list of sentences."""
    for sent in sent_list:
        if word_string in sent:
            return sent_list.index(sent)


# Set the few-shot training strings
default_fewshot = json_file
default_config = config_string

class GPTFeeder:
    def __init__(self, api_key=key, prompt_body="", prompt_pretext="", write_response_to_file=True,
                 json_filepath="fewshot.json", dont_train=False, config_filepath="config.cfg",
                 output_filepath="GPTResponses.txt"):
        """
        Note: You MUST have installed the spacy-llm library using *python -m pip install spacy-llm*
        in terminal BEFORE this. Documentation: https://github.com/explosion/spacy-llm

        Important Params:

        api_key: The API key, as a string. REQUIRED.

        prompt_body: A text (string) that you may want GPT to process. You do not need to set this at initialization;
        each model object can send indefinite number of prompts using the send_prompt() functionality. if a prompt_body
        is set at initialization, that will be what is sent to GPT if send_prompt() is called without any arguments.
        Note: by default, GPTFeeder will have GPT perform NER, based on fewshot.json training, on the prompt_body text,
        unless the config file is changed.

        prompt_pretext: If you want to tell GPT something before the main prompt body. Could be instructions/greeting
        like "Hello, here's your task." Same notes as prompt_body apply.

        json_filepath: The path to the file supposed to be used for few-shot training of GPT. Defaults
        to the default few-shot training file included with the code-package, which trains GPT
        *specifically* to recognize SF named entities. Adjust fewshot.json per your convenience.

        dont_train: If set to True, will not perform few-shot training.

        config_filepath: The path to the file used for Spacy configuration. Defaults to the config.cfg
        file already present in this base directory, which should again be customized for other projects.

        output_filepath: The file where GPT's responses will be stored, if write_response_to_file is true
        """

        # Check if there are too many words/tokens in the prompt
        prompt = prompt_body.split(" ")
        prompt_pre = prompt_pretext.split(" ")
        if len(prompt) + len(prompt_pre) > 3000:
            print("GPTFeeder does not support excessively large single prompts. Please reduce total prompt size"
                  " to less than 3000 words")
            return

        # Set API key
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.pipeline = None
        self.create_configuration(json_filepath, config_filepath)


    def create_configuration(self, json_filepath, config_filepath):
        """Configure GPTFeeder object's spacy-llm pipeline"""
        global default_fewshot
        global default_config

        json_dump = default_fewshot
        config_str = default_config

        with open(json_filepath, "w+") as outfile:
            json.dump(json_dump, outfile, indent=2)

        config = configparser.ConfigParser()
        config.read_string(config_str)

        with open(config_filepath, 'w+') as configfile:
            config.write(configfile)

        # Create the spacy pipeline with spacy-llm capability. Analogous to setting nlp = assemble("config.cfg")
        self.pipeline = assemble(config_filepath)
        # self.brief_test()

    def brief_test(self):
        """Test pipeline functionality"""
        nlp = self.pipeline
        doc = nlp("In the distant future, the visionary scientist Aurora Novak, known for revolutionary concepts like "
                  "Stellar Constructs")  # , partnered with the Galactic Innovators League to pioneer an advanced warp propulsion system. "
                  #"This warp engine, a marvel of futuristic engineering, enabled space travel through quantum tunnels, opening "
                  #"up new frontiers in the cosmos. Meanwhile, the Martian Homestead initiative")
        print("Text:", doc.text)
        print("Entities:", doc.ents)

    def send_prompt(self, text="", write_response_to_file=True, filepath="GPTResponses.json"):
        """Sends *text* as prompt to GPT (after GPT has been few-shot trained during configuration)
        and receives the response as a Spacy doc object, from which entities etc. can be extracted.
        By default, appends the response in json format to the default json-format file.

        Returns the response doc object also, so that user can manipulate it themselves if so desired."""
        response = self.pipeline(text)  # "Response" is a Spacy doc object
        out_dict = {"text": response.text,
                    "entities": [ent.text for ent in response.ents],
                    "entity_info": [(ent.text, ent.label_) for ent in response.ents]
                    }
        if write_response_to_file:
            with open(filepath, "a") as out_file:  # Append without overwriting
                json.dump(out_dict, out_file, indent=2)
        return response

if __name__ == '__main__':  # Set up the configuration and run a test

    os.environ["OPENAI_API_KEY"] = key  # VALID APY KEY HAS TO EXIST
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    our_json = default_fewshot
    our_config = default_config

    with open("fewshot.json", "w") as outfile:
        json.dump(json_file, outfile, indent=2)

    config = configparser.ConfigParser()
    config.read_string(config_string)

    with open("config.cfg", 'w') as configfile:
        config.write(configfile)

    # Create our model object
    model = GPTFeeder(key)

    # Bring in the list of sentences
    with open("sents_containing_named_entities.txt") as file:
        # Read the whole file as one string
        as_string = file.read()
        # Now split the file into sentences based on newlines
        sentences = as_string.splitlines()
        print("Check number of sentences:", len(sentences))
        print("First few:", sentences[:30])

    # index_where_story_starts = find_index_of_a_sentence(sentences, "police grilled him mercilessly")
    # print(index_where_story_starts, sentences[index_where_story_starts])

    # Weave our first paragraph using 531 sentences
    para = ". ".join(sentences[500:531])
    print(para)

    # Pass the para to the LLM and save its response as a Spacy doc object (using spacy-llm)
    doc = model.pipeline(para)

    output_dict = {"text": doc.text,
                   "entities": [ent.text for ent in doc.ents],
                   "entity_info": [(ent.text, ent.label_) for ent in doc.ents]
                   }
      #TODO: check if doc.text gives str or not

    with open("GPTResponses.json", "w+") as output_file:
        json.dump(output_dict, output_file, indent=2)

    #
    #

    #
    # nlp = assemble("../config.cfg")
    #
    # test_text = "Hello GPT-3.5, I'm am providing you a list of named entities. Can you please quickly run through it and pick" \
    #             "out the valid SF entities based on your training: " \
    #             "'joe harwitt', 'harwitt', 'lundgrens', 'westinghouse lundgrens', 'glad joker', 'carol lundgren', " \
    #             "'ellen nodded', 'hour bum', 'leslie seymour', 'camel', 'detail definitely disc', 'fifty gram', 'felipe camel', " \
    #             "'tenth kilometer', 'leslie sheepish', 'stared kevin', 'supply hour', 'drank coffee constantly', 'kevin trust', " \
    #             "'ellen cabin', 'bin telescope meter', 'ellen threw', 'like couple hundred meter', 'ellen kev', 'meteoroid afraid', " \
    #             "'kevin felt ellen sudden', 'pacificos', 'cry doom', 'soviet union realize development yea', 'rioter known', " \
    #             "'witness demonstration', 'livermore los alamo', 'freeman dyson long', 'nature kinder', 'degaulle give china atom bomb', " \
    #             "'aid turk', 'new berlin', 'soviet island', 'ussuri', 'west make', 'lin piao', 'lousiana', 'national space institute due', " \
    #             "'zisssuvzmysfis', 'landed upside', 'night day week', 'suggested press army government', 'hawkesworth', 'vic jennings', " \
    #             "'cynthia birthday', 'fourteen minute later', 'general nodded', 'jew harp', 'jennings fastball rock', 'moment moon'"
    #
    # # test_text = "In the distant future, the visionary scientist Aurora Novak, known for revolutionary concepts like " \
    # #             "Stellar Constructs, partnered with the Galactic Innovators League to pioneer an advanced warp propulsion system. " \
    # #             "This warp engine, a marvel of futuristic engineering, enabled space travel through quantum tunnels, opening " \
    # #             "up new frontiers in the cosmos. Meanwhile, the Martian Homestead initiative marked a pivotal moment in human " \
    # #             "history, with the idea of humans inhabiting Mars becoming a reality. As Martians reached the distant star " \
    # #             "Proxima Centauri, a mysterious and unexplored realm unfolded, showcasing the limitless possibilities of " \
    # #             "science fiction exploration."
    #
    # # test_text = "In the distant future, the visionary scientist Babu Sahab invented graviton toasters."
    #
    # doc = nlp(test_text)
    #
    # # Doc.text allows us to keep the original text so we can later manually assess the LLM's
    # # output
    # print(doc.text)
    #
    # # Doc.ents gives us the named entities that the LLM recognized
    # print(doc.ents)





