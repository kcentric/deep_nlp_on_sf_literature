# SNAPSHOT OF LLM INTERACTOR CODE BEFORE FIGURING OUT SPACY-LLM, TO POSSIBLY ILLUSTRATE THE DIFFERENCE
# THE LIBRARY FINALLY MADE.

# from main import api_key
from typing import Iterable, List
import json
import openai
import pandas as pd
import os
import spacy  # NOTE: Install spacy-llm FIRST. Documentation: https://github.com/explosion/spacy-llm
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.util import split_labels
import os
import json
import configparser
from spacy_llm.util import assemble
from datasets import load_dataset


labels = []


def read_named_entities_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    named_entities_list = df['NamedEntities'].tolist()
    return named_entities_list


@registry.llm_tasks("my_namespace.MyTask.v1")
def make_my_task(labels: str, my_other_config_val: float) -> "MyTask":
    labels_list = split_labels(labels)
    return MyTask(labels=labels_list, my_other_config_val=my_other_config_val)


class MyTask:  # Added more rigorous parameters in this class because began gaining
    # a deeper understanding of them by reading Argilla and Spacy documentation and code
    def __init__(self, labels: List[str], my_other_config_val: float):
        ...

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        ...

    def parse_responses(
            self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        ...


class GPTFeeder:
    def __init__(self, api_key, prompts, data, data_preprocessed=True, chunksize=2000,
                 write_responses_to_file=True):
        """Takes in a list of prompts to be fed sequentially to GPT model (prompts will be fed
        in an order corresponding to the order of data).

        Note: You MUST have installed the spacy-llm library using *python -m pip install spacy-llm*
        in terminal BEFORE this. Documentation: https://github.com/explosion/spacy-llm

        Important Params:
        - api_key: The API key, as a string. REQUIRED.
        - prompts: The list of prompts to be fed to GPT, one by one.
        - chunksize: refers to number of elements to feed GPT per prompt; maxes out at 2500 tokens.
        - data: A list of strings. *chunksize* elements of this list will be fed to GPT with each
        prompt. For example: If the first prompt is "Here is the first four Named Entities from
        my dataset", and *chunksize* is 4, then the prompt fed to GPT will be "Here is ... dataset.
        [Entity 1], [Entity 2] ..." etc.

        If *data_preprocessed* is False, will process data appropriately before feeding to GPT.
        """
        # Set API key
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.data = data
        self.prompts = prompts
        self.chunksize = chunksize
        self.llm = spacy.load("en_core_web_lg").add_pipe("llm_ner")
        self.do_the_talking(data_preprocessed, write_responses_to_file)

    # TODO: Complete this function
    def get_gpt_insights(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text.strip()

    # TODO: Complete this function
    def interact_with_gpt(self, prompts, csv_path):
        named_entities_list = read_named_entities_from_csv(csv_path)

        for prompt, named_entities in zip(prompts, named_entities_list):
            # Format the prompt for GPT-3.5
            gpt_prompt = f"{prompt} The named entities in the text are: {named_entities}"

            # Get insights from GPT-3.5
            gpt_insights = self.get_gpt_insights(gpt_prompt)

            # Print results
            print(f"Prompt: {prompt}")
            print(f"Named Entities: {named_entities}")
            print(f"GPT-3.5 Insights: {gpt_insights}")
            print("\n")

    def do_the_talking(self):
        pass


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "sk-abuchakaabuabu2222"  # INSERT YOUR OWN VALID API KEY HERE
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    json_file = [
        {
            "text": "I was charged with an exchange rate for my purchase and it was not right.",
            "spans": [
                {
                    "text": "charged",
                    "is_entity": 'false',
                    "label": "==NONE==",
                    "reason": "is an action done to the customer by the bank, not by them"
                },
                {
                    "text": "purchase",
                    "is_entity": "true",
                    "label": "ACTIVITY",
                    "reason": "is an action that the customer did, not the bank"
                }
            ]
        },
        {
            "text": "The exchange rate for the last item I bought seems to be correct.",
            "spans": [
                {
                    "text": "exchange rate",
                    "is_entity": "false",
                    "label": "==NONE==",
                    "reason": "is a name for currency, not an action or performance"
                },
                {
                    "text": "item",
                    "is_entity": "false",
                    "label": "==NONE==",
                    "reason": "is a generic name for the object bought, not a performance"
                },
                {
                    "text": "bought",
                    "is_entity": "true",
                    "label": "ACTIVITY",
                    "reason": "is the verb for the action that the customer performed"
                }
            ]
        }
    ]

    config_string = """
      [paths]
      examples = "fewshot.json"

      [nlp]
      lang = "en"
      pipeline = ["llm","sentencizer"]

      [components]

      [components.llm]
      factory = "llm"

      [components.llm.task]
      @llm_tasks = "spacy.NER.v3"
      labels = ["PERSON", "ORGANIZATION", "CARDINAL", "PERCENT", "ACTIVITY"]

      [components.llm.task.examples]
      @misc = "spacy.FewShotReader.v1"
      path = "${paths.examples}"

      [components.llm.model]
      @llm_models = "spacy.GPT-3-5.v1"
      config = {"temperature": 0.5}

      [components.sentencizer]
      factory = "sentencizer"
      punct_chars = null
    """

    with open("../LLM/fewshot.json", "w") as outfile:
        json.dump(json_file, outfile, indent=2)

    config = configparser.ConfigParser()
    config.read_string(config_string)

    with open("../main files/config.cfg", 'w') as configfile:
        config.write(configfile)

    nlp = assemble("../main files/config.cfg")
    doc = nlp("The same item is 50% more expensive now, so they did not purchase it.")

    # Doc.text allows us to keep the original text so we can later manually assess the LLM's
    # output
    print(doc.text)

    # Doc.ents gives us the named entities that the LLM recognized
    print(doc.ents)





