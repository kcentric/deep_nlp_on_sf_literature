# Deep NLP: SF Literature Analysis

## Introduction

Hello! Welcome to my debut project. The [dataset](https://www.kaggle.com/datasets/jannesklaas/scifi-stories-text-corpus/data) I began with for this project, sourced largely from the [Pulp Magazine Archive](https://archive.org/details/pulpmagazinearchive), is a huge collection of science fiction stories in a single-file text corpus, 149.33MB in raw size. Here's how the first couple lines look in PyCharm 🙂 (it's the editor's intro to [IF Magazine](https://archive.org/details/ifmagazine)):

<img width="1142" alt="Screenshot 2024-01-24 at 4 49 11 PM" src="https://github.com/kkrishna24/deep_nlp_on_sf_literature/assets/121068842/05e73979-e2cb-4baf-9dd6-1280d34aab43">

The stories span multiple decades and contain a variety of writing styles, themes, and ideas. They represent a good snapshot of 20th-century SF literature, and have demonstrated their usefulness before for awesome projects like [Robin Sloan's Autocomplete](https://www.robinsloan.com/notes/writing-with-the-machine/).

I wanted to analyze the corpus itself, and in the process gain insights into the era of SF literature it represents. I decided to use a multi-pronged, multi-stage approach, in each step focusing on making my code as generalizable and well-documented as possible. The steps were as follows:
- **Step 1: Rigorous preparation** of the data. You can find the code for this in `CorpusProcessor_with_NER.py`, [here](https://github.com/kkrishna24/deep_nlp_on_sf_literature/blob/main/main%20files/CorpusProcessor_with_NER.py).
  - I split the text into units of sentences and tokens using [NLTK](https://www.nltk.org/). Tokens are analogous to words in that they are units of meaning: they make text better suited for Natural Language Processing (NLP). Here are some [examples](https://www.nltk.org/howto/tokenize.html).
  - I **clean** the tokenized text by removing spaces, words like "a, an, the" that may not add meaning but are very frequent in the data, etc. The function doing most of this work is `clean_string`. Its header looks like this in `CorpusProcessor`:

    ```python
      def clean_string(self, text, only_remove_line_breaks=False, pos_tokens_if_lemmatizing=None, find_pos=False,
                       stem="None", working_on_stories=True):
          """Takes in a string and removes line breaks, punctuation, stop-words, numbers, and proceeds to stem/lemmatize.
          Returns the "cleaned" text finally. Capable of nuances depending on inputs."""
    ```
     Note that we have a `working_on_stories` parameter that defaults to `True`. The reason I made my cleaning methods so versatile and customized is that for domain-specific datasets, merely removing "a, an, the" etc. **is not sufficient**. For example, the words "said," "replied," "asked" might be just as common in a story-corpus as the article "and". (I have not statistically modeled this. Feel free to drop me a comment about whether this is true 🙂)
  - Besides cleaning, CorpusProcessor's `initial_prep` also [lemmatizes](https://builtin.com/machine-learning/lemmatization) the text, and then creates a representation of the whole corpus as a (humongous) list of words, as a list of sentences, and as a list of cleaned sentences.

    ```python
    self.text_as_a_wordlist = list(itertools.chain.from_iterable(list_of_lists))  # Now we can easily use this wordlist
                                                                                      # for further processing
    self.list_of_sentences_original = list_of_sentences
    self.list_of_sentences_cleaned = cleaned_sentences
    ```
    By storing these various representations of the text in attributes, the `CorpusProcessor` object will then allow us to flexibly perform numerous tasks with the text as our need be. Indeed, as `initial_prep` itself shall summarize for you, if you are mistaken enough to try get a return value from it ... 🙂

    ```python
    # Having some fun :)
    return "This function does not return prepped text, but rather just preps the text to now be contained" 
           "as the CorpusProcessor's attributes. Please call those attributes if you wish to see the cleaned text :-)"
    ```
    The main functions that I use CorpusProcessor's attributes for are to split the corpus into individual books (by splitting around valid appearances of 'Copyright' in the wordlist) so that I can have separate documents for topic-modeling via [LDA](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html), and to create a Pandas dataframe containing the 3.5 million (cleaned) sentences of the corpus: the dataframe is very useful for [NER](https://spacy.io/usage/spacy-101).

- **Step 2: Multi-stage named entity recognition** (NER). This turned out to be the most involved part of the project. You can begin exploring in `NER.py`, [here](https://github.com/kkrishna24/deep_nlp_on_sf_literature/blob/main/main%20files/NER.py).
  - [NER](https://www.turing.com/kb/a-comprehensive-guide-to-named-entity-recognition) is typically the process of extracting words or series of words from a text and categorizing them under common labels like "Person," "Organization," "Location," or custom labels like "Healthcare Terms," "Programming Languages" etc. For my corpus, I was primarily interested in extracting terms that represented *SF technologies*, *SF concepts*, and miscellaneous significant terms that contained plot- or theme-related meaning.
  - I began by performing a blanket-NER task across the whole corpus, using a Pandas dataframe created earlier from `CorpusProcessor` as my input. The Pandas dataframe contained the corpus as a list of sentences, and my optimized NER algo using the [spaCy](https://spacy.io/api/entityrecognizer) library with its parallelization capabilities generated a **new** dataframe that would look like:
    
    ```python
                                              Sentence          Entity Entity_type
    0  In a galaxy far, far away, Luke Skywalker tra...  Luke Skywalker      Person
    1  The year 2050 marked a new era for humanity.                2050        Date
      ```

The aims of this project are threefold:
- Develop a repository that **easily** be **refitted, reused, or expanded** to many different kinds of domain-specific literature analysis tasks. This is the overall, community-oriented guiding vision.
- Conduct a multifarious analysis of a corpus of science fiction (SF) literature spanning several decades of the 20th century, in order to extract insights about SF literature in general, and to then present the results as visually and clearly as possible. This is the immediate purpose of my project and what the code as-is accomplishes most effectively.
- To perform this analysis, and design the code performing each stage of it, as efficiently as possible so as to both reduce project dependency on proprietary external services, and conserve a user's budget as much as possible. This is the user-centric part of my vision.

### Note

This project is in its final stages of development. Most individual modules are complete and can be used or repurposed. The documentation is also comprehensive and mostly up-to-date. However, please know that some of the organization and in-project README.txt files have not been updated yet. For example, the README.txt in the **[Archive](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/Archive)** directory does not entirely capture how the directory has evolved since then. 

As of this writing, I am currently in the process of [fine-tuning a RoBERTa model](https://colab.research.google.com/drive/1Md5Lpe4WiLxeDUzxZk8vdkRmA50MIUMh?usp=sharing) with the output acquired from `GPT_NER_Round_1.py`, in preparation for performing last-stage NER using [RoBERTa](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md). If you wish to explore the project code, I suggest beginning with `CorpusProcessor_with_NER.py` in **[main files](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/main%20files)** (which actually does not contain any NER implementation within itself yet, despite being fully functional otherwise). There you will find code for multifarious data preprocessing. If you wish to run it as a script, however, please set the `FILEPATH` variable to a file you have access to and want to process. I have not hosted "internet_archive_scifi_v3.txt" (the corpus file I was using) on GitHub yet, thanks to its size.

## Overview and guide

Most of the action as of the current version is contained in the directories **[main files](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/main%20files)**, **[TF-IDF](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/TF-IDF)**, and **[LLM](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/LLM)**. In **main files**, begin with `CorpusProcessor_with_NER.py` that was used for preprocessing the corpus. Then go to `NER.py` that performs first-stage NER on the corpus using [spaCy](https://github.com/explosion/spaCy) and stores the output in "NER_output2.csv" by default.

After performing basic NER, you can check out `sentences_of_entities.py`. If you run it as a script, it (attempts to) extract the famed 3.5M sentences from "sci-fi_text_cleaned.csv", the csv file representing a [Pandas dataframe](https://github.com/pandas-dev/pandas) that contains the corpus as a list of sentences with their cleaned versions. Then from this list, `sentences_of_entities` extracts all the sentences that contain any of the entities that were found by spaCy's NER. (Note that these entities are brought in from "entity_list.txt" which is a file I created using a separate script, after running `NER.py`. "entity_list.txt" _is_ small enough to be included in the "main files" directory, so it is. Also note that in the current version of the code, "sci-fi_text_cleaned.csv" has not been added already to **main files**. You may have to create this file or slightly modify the code in `sentences_of_entities` to have it work as a script)

Now the results of `sentences_of_entities.py` will be stored in the **LLM** directory, in "sents_containing_named_entities.txt". But don't go to the **LLM** directory just yet: first go to **TF-IDF** and run `TF-IDF.py` as a script. This will use [scikit-learn](https://github.com/scikit-learn/scikit-learn)'s [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and KMeans clustering to extract a **diverse** batch of 8000 sentences from `sents_containing_named_entities.txt`. The diverse batch of sentences will be stored in `representative_sents_for_GPT_labelling.txt` that remains in the `TF-IDF` directory. Why shall we be using a representative sample rather than the full 114,000+ entity-containing sentences? Because otherwise, we'll need a _lot_ of [OpenAI API](https://github.com/openai/openai-python) tokens to run a round of custom NER using [GPT-3.5 Turbo](https://platform.openai.com/docs/guides/text-generation/chat-completions-api).

Finally, you can now go to the directory **LLM**, open `GPT_NER_round_1.py`, and run it as a script. I have already created data you can use, so when asked about it you can use any key except "Y" to generate RoBERTa training data. If you want to see real magic however, first go to `api_key.py` and insert a valid OpenAI API key. Then run `GPT_NER_round_1` again and say "Y" when it asks you for input. You can then see as we use [spaCyLLM](https://github.com/explosion/spacy-llm) and [OpenAI API](https://github.com/openai/openai-python) to label and extract terms from the text as **TECHNOLOGY**, **CONCEPT**, or **MISCELLANEOUS SIGNIFICANT**. Then, we [tokenize](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py) this output appropriately and prepare to use it for [fine-tuning a RoBERTa model](https://colab.research.google.com/drive/1Md5Lpe4WiLxeDUzxZk8vdkRmA50MIUMh?usp=sharing), so RoBERTa can perform the remainder of our required NER - for free.

**Note on 'LLM'**: `LLM_Interactor.py` is the module that handles the creation of - you guessed it - an LLM interaction object. It configures spaCyLLM - [please check the documentation](https://spacy.io/api/large-language-models) - using the configuration strings residing in `config_strings_2.py`, which in turn refers to `fewshot.json` for its [fewshot-training](https://blog.paperspace.com/few-shot-learning/) info. **This means that `LLM_Interactor` as well as `GPT_NER_round_1` are very independent and malleable modules.** By modifying `fewshot.json`, you can train GPT-3.5 to perform NER for any other kind of domain-specific task. You can also change the spacyLLM configuration by modifying `config_strings_2.py`, where you can modify the "labels" to be fed for your own domain-specific task. For example, you can change these lines:

```python
`[components.llm.task]`  
`@llm_tasks = "spacy.NER.v3"`
`labels = ["TECHNOLOGY", "CONCEPT", "MISCELLANEOUS SIGNIFICANT"]`
```
to these lines:

```python
  `[components.llm.task]`
  `@llm_tasks = "spacy.NER.v3"`
  `labels = ["FINANCIAL ENTITY", "NON-FINANCIAL ENTITY"]`
```
if you were doing NLP on finance-related texts. You can even change the LLM model used to other ones supported by spaCyLLM.

## Features to come

- After RoBERTa-powered NER, we shall be using `gensim` [LDA](https://radimrehurek.com/gensim/models/ldamodel.html) and `scikit-learn` to model topics, themes etc. for further analysis on this "substantial essence" (represented by tech, concept and character-containing sentences) of the corpus.
  - We are treating the sentences containing prominent domain-specific named-entities to be the essence of our corpus that we're concerned with. However, depending on the particular analysis _you_ want to perform, you might have extracted sentences containing other kinds of terms/entities as your essence. For example, if you wanted to model the _social_ realism of an SF corpus rather than its _scientific_ realism, or if you were examining the works of an author like [Octavia E. Butler](https://www.octaviabutler.com/theauthor), you might prefer to extract various gender-specific or worldbuilding-specific terms. Our corpus contains works more like [Isaac Asimov](https://www.britannica.com/biography/Isaac-Asimov)'s, which tend to be idea-heavy (i.e. tech and concept heavy). Thus it is reasonable to consider "tech" and "concept" terms are representing their essence, with "miscellaneous significant" capturing other theme or plot-important terms.
  - You might also model the "essence" of your corpus in another way altogether (for example, using sentiment analysis rather than NER). 
- We shall be using time-series analysis etc. to model the interplay of the concepts and tech-terms mentioned in the corpus. Our custom-NER output gives us a lot of leeway to examine the concepts that drive the SF corpus. Besides getting insights about how SF concepts change/evolve over the corpus, we might also get insights on how those concepts changed/evolved in actual real-world progression through the 20th century. (For example, if concepts do change over the corpus, that might simultaneously be a sign that the corpus itself represents stories that were written in a linear progression through time in the 20th century) This is likely because I am already aware that the corpus begins with the first issue of [the 'IF' magazine](https://gizmodo.com/the-entire-run-of-if-magazine-is-now-freely-available-o-1761691317) and several later texts are chronologically-later issues of the same magazine. However, after we're done with 'IF' magazine's issues, the text contains [Galaxy Magazine](https://www.theverge.com/2017/7/14/15970710/galaxy-science-fiction-magazine-online-free-reading-archive)'s early issues, which might reset our clock.
- A final `main.py` file will be created which will contain code to visualize our results. This shall complete the project.

For more about me, see my [Linkedin profile](https://www.linkedin.com/in/krishnatripathi070/).
