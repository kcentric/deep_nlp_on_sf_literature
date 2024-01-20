# Strings for simplified configuration settings. Few-shot training and config will center around
# dividing entities into two types only: "SF" and "Non-SF" (Entity) with the rest being "None"
#TODO: Maybe need to add "None"?

# This few-shot string does not have any is_entity criterion: all spans are classified as either SF or non-SF
json_file = [
    {
        "text": "and freeman dyson of who lived in new york studied hairless bipeds which walked upright",
        "spans": [
            {
                "text": "freeman dyson",
                "is_entity": "true",
                "label": 'SF Entity',
                "reason": "is a scientist associated with the concept of dyson spheres, an important "
                          "science fiction concept"
            },
            {
                "text": "new york",
                "is_entity": "true",
                "label": "Non-SF",
                "reason": "is a real-world city, not an SF concept or term"
            },
            {
                "text": "hairless bipeds",
                "label": "SF Entity",
                "reason": "a distinctly SF way of referring to humans"
            }
        ]
    },
    {
        "text": "new people's asia invented the warp drive, enabling hyperspace donut",
        "spans": [
            {
                "text": "warp drive",
                "label": "SF Entity",
                "reason": "is the name of a common SF transportation technology"
            },
            {
                "text": "new people's asia",
                "label": "SF Entity",
                "reason": "is the name of a fictional political entity, likely important to the story"
            },
            {
                "text": "hyperspace",
                "label": "SF Entity",
                "reason": "a commonly used concept and term in SF"
            },
            {
                "text": "donut",
                "label": "Non-SF",
                "reason": "does not in itself constitute an SF concept, though in combination with others "
                          "it could"
            }
        ]
    },
    {
        "text": "a year after venusian colonization, venusians reached epsilon eridani",
        "spans": [
            {
                "text": "venusians",
                # "is_entity": "false",
                "label": "SF Entity",
                "reason": "most likely the name of a species of humans or aliens"
            },
            {
                "text": "venusian colonization",
                # "is_entity": "true",
                "label": "SF Entity",
                "reason": "refers to the idea or theoretical possibility of human colonization on Venus"
            },
            {
                "text": "epsilon eridani",
                # "is_entity": "true",
                "label": "SF Entity",
                "reason": "name of a star commonly referred to in SF, potentially representing an interstellar civilization "
                          "or significant location in the text"
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
  labels = ["SF Entity", "Non-SF"]

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
