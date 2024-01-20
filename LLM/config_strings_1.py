# Default strings for few-shot training and spacy-llm config

json_file = [
    {
        "text": "and freeman dyson of princeton institute for advanced studies.",
        "spans": [
            {
                "text": "freeman dyson",
                "is_entity": 'true',
                "label": 'PERSON',
                "reason": "is a scientist associated with the concept of dyson spheres, an important "
                          "science fiction concept"
            },
            {
                "text": "princeton institute for advanced studies",
                "is_entity": "false",
                "label": "==NONE==",
                "reason": "is a real-world academic institution, not an SF concept"
            }
        ]
    },
    {
        "text": "new people's asia invented the warp drive, enabling hyperspace donut",
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
                "label": "ORGANIZATION",
                "reason": "is the name of a political entity"
            },
            {
                "text": "hyperspace donut",
                "is_entity": "false",
                "label": "==NONE==",
                "reason": "a whimsical fusion of concepts, not recognized in any scientific or common context"
            }
        ]
    },
    {
        "text": "a year after venusian colonization, venusians reached epsilon eridani",
        "spans": [
            {
                "text": "venusians",
                "is_entity": "false",
                "label": "==NONE==",
                "reason": "name of a species of humans or aliens, not a technology, concept or notable element"
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
                "label": "OTHER SF",
                "reason": "name of a star commonly referred to in SF, potentially representing a relevant location "
                          "in the text, but not fitting in any of the other categories"
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
  labels = ["CONCEPT", "ORGANIZATION", "TECHNOLOGY", "PERSON", "OTHER SF", "==NONE=="]

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
