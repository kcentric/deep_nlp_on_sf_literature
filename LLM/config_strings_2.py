# Default strings for few-shot training and spacy-llm config

json_file = [
    {
        "text": "and freeman dyson of princeton institute for advanced studies, who lived in new york, "
                "studied hairless bipeds which walked upright, using microscopes",
        "spans": [
            {
                "text": "freeman dyson",
                "is_entity": 'true',
                "label": 'MISCELLANEOUS SIGNIFICANT',
                "reason": "is a scientist associated with the concept of dyson spheres, an important idea in "
                          "science fiction"
            },
            {
                "text": "princeton institute for advanced studies",
                "is_entity": "true",
                "label": "MISCELLANEOUS SIGNIFICANT",
                "reason": "is an academic institution connected with scientific advancements "
                          "and science fiction ideas"
            },
            {
                "text": "new york",
                "is_entity": "false",
                "label": "==NONE==",
                "reason": "is a real-world city rarely linked directly with science fiction ideas"
            },
            {
                "text": "hairless bipeds",
                "is_entity": "true",
                "label": "CONCEPT",
                "reason": "is a distinctly science fiction way of referring to humans"
            },
            {
                "text": "microscopes",
                "is_entity": "true",
                "label": "TECHNOLOGY",
                "reason": "are a well-established technology"
            }
        ]
    },
    {
        "text": "new people's asia invented the warp drive, enabling people to eat donuts in hyperspace rings",
        "spans": [
            {
                "text": "warp drive",
                "is_entity": "true",
                "label": "TECHNOLOGY",
                "reason": "is the name of an SF transportation technology"
            },
            {
                "text": "new people's asia",
                "is_entity": "true",
                "label": "MISCELLANEOUS SIGNIFICANT",
                "reason": "is the name of a fictional political entity, of a kind which is common in"
                          " science fiction"
            },
            {
                "text": "hyperspace rings",
                "is_entity": "true",
                "label": "CONCEPT",
                "reason": "a term that is likely a story-specific extension of the common SF concept of hyperspace"
            }
        ]
    },
    {
        "text": "a year after venusian colonization, venusians reached epsilon eridani, with damien kirk guiding the ship "
                "along with the help of his lieutenants Valerie, Dakin and others",
        "spans": [
            {
                "text": "venusians",
                "is_entity": "false",
                "label": "==NONE==",
                "reason": "name of a species of humans or aliens, though not a technology or concept"
            },
            {
                "text": "venusian colonization",
                "is_entity": "true",
                "label": "CONCEPT",
                "reason": "refers to the idea or theoretical possibility of human colonization on Venus"
            },
            {
                "text": "epsilon eridani",
                "is_entity": "true",
                "label": "MISCELLANEOUS SIGNIFICANT",
                "reason": "name of a star commonly referred to in SF, potentially representing a relevant location "
                          "in the text, though not fitting in any of the other categories"
            },
            {
                "text": "damien kirk",
                "is_entity": "true",
                "label": "MISCELLANEOUS SIGNIFICANT",
                "reason": "in context, since this character is guiding the ship it must be a significant character"
            },
            {
                "text": "Valerie",
                "is_entity": "false",
                "label": "==NONE==",
                "reason": "is not a significant character because is only mentioned once, and in context is merely "
                          "a supporting character"
            }
        ]
    }
]

# NOTE: Removed "PERSON" from labels list
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
  labels = ["TECHNOLOGY", "CONCEPT", "MISCELLANEOUS SIGNIFICANT"]

  [components.llm.task.examples]
  @misc = "spacy.FewShotReader.v1"
  path = "${paths.examples}"

  [components.llm.model]
  @llm_models = "spacy.GPT-3-5.v2"
  config = {"temperature": 0.5}

  [components.sentencizer]
  factory = "sentencizer"
  punct_chars = null
"""
