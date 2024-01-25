# Deep NLP: SF Literature Analysis

## Introduction 👋

The [dataset](https://www.kaggle.com/datasets/jannesklaas/scifi-stories-text-corpus/data) I began with for this project, sourced largely from the [Pulp Magazine Archive](https://archive.org/details/pulpmagazinearchive), is a huge collection of science fiction stories in a single-file text corpus, 149.33MB in raw size. Here's a [link](https://issuu.com/565585/docs/1952__03__march___if) to the first book in the corpus, and here's a snapshot of how the text looks in PyCharm: 

<img width="881" alt="Screenshot 2024-01-25 at 10 07 15 AM" src="https://github.com/kkrishna24/deep_nlp_on_sf_literature/assets/121068842/d8275325-3377-4f9d-ae6a-4a8cbf535eee">

The stories span multiple decades and contain a variety of writing styles, themes, and ideas. They represent a good snapshot of 20th-century SF literature, and have demonstrated their usefulness before for awesome projects like [Robin Sloan's Autocomplete](https://www.robinsloan.com/notes/writing-with-the-machine/).

I wanted to analyze the corpus itself, and in the process gain insights into the era of SF literature it represents. I decided to use a multi-pronged, multi-stage approach, in each step focusing on making my code as generalizable and well-documented as possible. 

### Key Highlights
- Processed texts using customized methods, [NLTK](https://www.nltk.org/), and [spaCy](https://spacy.io/)
- Performed domain-specific named entity recognition in multiple stages
- Fine-tuned a [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) model using [GPT](https://platform.openai.com/docs/models) to generate annotated data
- Implemented [multicore LDA](https://radimrehurek.com/gensim/models/ldamulticore.html) for efficient topic modeling and theme-extraction
- Modularized code to make it highly reusable for other domain-specific literature tasks: code can be easily refitted for legal datasets, a corpus of classics etc.

An example of GPT-generated annotated data (ain't it interesting? 🙂):

```python
[('buddha', 'MISCELLANEOUS SIGNIFICANT'), ('pistol shots', 'TECHNOLOGY'), ('chart', 'TECHNOLOGY'),
('torpedo', 'TECHNOLOGY'), ('steel shelves', 'TECHNOLOGY'), ('southeast asia', 'MISCELLANEOUS SIGNIFICANT')]
```

Some basic NER output:

```python
                                                               Sentence         Entity   Entity_type
2294  plane cloud degree climb dozen mile towards philippine anyone yat...    dozen mile    QUANTITY
2295                                                    seen kyoto buddha          kyoto         GPE
2296                                 would make building quarter mile long  quarter mile    QUANTITY
```
An example of a cleaned string vs an uncleaned string:

```python
     Sentences                                                Cleaned Sentences
122, yet in the way they moved and in the way they stood      yet way moved way stood
```

### How to use your own corpus

#### Step 0:
- Insert a plain text file in the `Input Files` directory. There is no specific format it has to be in, but it works best if it's as regular a text file as we get. Check out [`small_sample_text_for_testing.txt`](https://github.com/kkrishna24/deep_nlp_on_sf_literature/tree/main/Input%20Files) to see how your corpus/file should look like.
- Go to `CorpusProcessor_with_NER.py`. Change the `FILEPATH` variable, right at the top of the code, to say "../Input Files/`[your_file_name]`.txt". Replace `your_file_name` with your actual file name.
- Make sure to uncomment the line saying `nltk.download()` right above `FILEPATH`, if you don't have NLTK on your system.
  Here's how the code-snippet at the top of the `CorpusProcessor` module should like then:
  
  <img width="1005" alt="Screenshot 2024-01-25 at 10 40 13 AM" src="https://github.com/kkrishna24/deep_nlp_on_sf_literature/assets/121068842/f62fa394-3d24-4141-98e8-00b7d6d2f321">

  Note how `nltk.download()` is not commented out now.

#### Step 1 🚀:
- And that's it! You're ready to go. Run `CorpusProcessor_with_NER` as a script, and see your data get cleaned 🧼, ready for further processing. Then follow the [exploration guide](#exploration-guide) below.

## Overview of the project
The steps I went through were as follows:

### Step 1: Rigorous data preparation 
You can find the code for this in `CorpusProcessor_with_NER.py`, [here](https://github.com/kkrishna24/deep_nlp_on_sf_literature/blob/main/main%20files/CorpusProcessor_with_NER.py).
  - I split the text into units of sentences and tokens using [NLTK](https://www.nltk.org/). Tokens are analogous to words in that they are units of meaning: they make text better suited for Natural Language Processing (NLP). Here are some [examples](https://www.nltk.org/howto/tokenize.html).
  - I **clean** the tokenized text by removing spaces, words like "a, an, the" that may not add meaning but are very frequent in the data, etc. The function doing most of this work is `clean_string`. Its header looks like this in `CorpusProcessor`:

    ```python
    def clean_string(self, text, only_remove_line_breaks=False, pos_tokens_if_lemmatizing=None, find_pos=False,
                     stem="None", working_on_stories=True):
        """Takes in a string and removes line breaks, punctuation, stop-words, numbers, and proceeds to stem/lemmatize.
        Returns the "cleaned" text finally. Capable of nuances depending on inputs."""
    ```
     Note that we have a `working_on_stories` parameter that defaults to `True`. The reason I made my cleaning methods so versatile and customized is that for domain-specific datasets, merely removing "a, an, the" etc. **is not sufficient**. For example, the words "said," "replied," "asked" might be just as common in a story-corpus as the article "and". (I have not statistically modeled this. Feel free to drop me a comment about whether this is true 🙂)
  - Besides cleaning, CorpusProcessor's `initial_prep` also [lemmatizes](https://builtin.com/machine-learning/lemmatization) the text, then generates representations of the corpus as a (humongous) list of words, as a list of sentences, and as a list of cleaned sentences. Code snippet:

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

### Step 2: Multi-stage named entity recognition (NER) 
This turned out to be the most involved part of the project. You can begin exploring in `NER.py`, [here](https://github.com/kkrishna24/deep_nlp_on_sf_literature/blob/main/main%20files/NER.py).
  - [NER](https://www.turing.com/kb/a-comprehensive-guide-to-named-entity-recognition) is typically the process of extracting words or series of words from a text and categorizing them under common labels like "Person," "Organization," "Location," or custom labels like "Healthcare Terms," "Programming Languages" etc. For my corpus, I was primarily interested in extracting terms that represented *SF technologies*, *SF concepts*, and miscellaneous significant terms that contained plot- or theme-related meaning.
  - I began by performing a blanket-NER task across the whole corpus, using a Pandas dataframe, created earlier by `CorpusProcessor`, as my input. The Pandas dataframe contained the corpus as a list of sentences, and my optimized NER algo using the [spaCy](https://spacy.io/api/entityrecognizer) library with its parallelization capabilities generated a **new** dataframe that would look like:
    
    ```python
                                              Sentence          Entity Entity_type
    0  galaxy far, far away, Luke Skywalker tra...       Luke Skywalker      Person
    1  year 2050 marked new era humanity.                          2050        Date
      ```
    This gave me a lot of meaning-containing sentences with their general entities. `spaCy` by itself of course cannot do something as specialized as extracting SF technology- and concept-terms from the text (without a significant amount of training, at least), which took me to the next stage.
  - I prepared a module (check [`sentences_of_entities.py`](https://github.com/kkrishna24/deep_nlp_on_sf_literature/blob/main/main%20files/sentences_of_entities.py)) to help me extract all _original_ (uncleaned/unmodified) sentences from the corpus which contain _any_ of the named entities found by spaCy: essentially, the original versions of the sentences which I fed to spaCy. For example, the above dataframe would now be modified to look like:
    
    ```python
                                              Sentence          Entity Entity_type
    0  In a galaxy far, far away, Luke Skywalker tra...  Luke Skywalker      Person
    1  The year 2050 marked a new era for humanity.                2050        Date
      ```
    Note the return of words like "a", "the", "in", "for" etc., which for the previous stage had been removed as [stop-words](https://kavita-ganesan.com/what-are-stop-words/).
    - The reason for this is that I planned to integrate with an LLM (large language model) like [GPT-3.5](https://platform.openai.com/docs/models) for some of my next steps, and LLMs are designed to generally do _better_ on texts which look exactly like what a human would write, than bare-bones word-strings.
  - At this point, I had a set of ~135k sentences. I further used [scikit-learn](https://github.com/scikit-learn/scikit-learn)'s [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and KMeans clustering to extract a diverse batch of ~10k sentences to represent this set. 
    - Find the TF-IDF code in [`TF-IDF`](https://github.com/kkrishna24/deep_nlp_on_sf_literature/tree/main/TF-IDF).
    - The reason we want to use ~10k sentences rather than ~135k (or 3.5 million, for that matter) is that we would have to _pay_ for that API usage (OpenAI's software is proprietary) - not to mention the huge increases in processing time that come with talking with an LLM in real-time (it took ~2 hours to run 10k sentences past GPT-3.5! That was the longest processing time for any single stage of my code).
  - I then ran `GPT_NER_Round_1.py`, which you can find in [this](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/LLM) directory along with its prep and configuration modules. `GPT_NER_Round_1` integrates few-shot learning with [`spacy-llm`](https://spacy.io/api/large-language-models) and OpenAI API to extract terms under `TECHNOLOGY`, `CONCEPT`, and `MISCELLANEOUS SIGNIFICANT` labels from our 10k-sentence sample dataset. The output is stored in `GPTResponses.json`, which is a list of Python dictionaries, each in this basic format:

  ```python
  {
    "span_of_para_in_sentence_list": [ "Beginning:0", "End:10" ],
    "entities_in_para": [ [ "second class matter", "CONCEPT" ], [ "howard browne", "MISCELLANEOUS SIGNIFICANT" ] ]
  }
```
  - After getting this GPT-generated data, I now use it as [training data](https://discuss.huggingface.co/t/training-a-domain-specific-roberta-from-roberta-base/2324) for [fine-tuning a RoBERTa model](https://colab.research.google.com/drive/1Md5Lpe4WiLxeDUzxZk8vdkRmA50MIUMh?usp=sharing). RoBERTa, unlike GPT-3.5, is an [open-source model](https://www.ibm.com/blog/open-source-large-language-models-benefits-risks-and-types/) which is perfectly suited for my goal of achieving project independency and reducing budget as much as possible. Find RoBERTa documentation [here](https://huggingface.co/docs/transformers/model_doc/roberta).
  - The fine-tuned RoBERTa model then performs the remainder of our SF technology-, concept- and miscellaneous-term extraction for our ~135k entity-containing sentences from across the corpus.

The next steps, Step 3 and Step 4, are "features to come"; I am currently in the process of completing my RoBERTa fine-tuning.

### Step 3: LDA topic modeling, time series analyses, and general collation
  - After RoBERTa-powered NER, we shall be using [Latent Dirichlet Allocation](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2) via the [`gensim`](https://radimrehurek.com/gensim/models/ldamodel.html) library, along with [`scikit-learn`](https://scikit-learn.org/stable/) to model topics, themes etc. for further analysis on this "substantial essence" (represented by tech, concept and character-containing sentences) of the corpus.

### Step 4: Creation of final graphs and visualization of results, final commit
  - An `app.py` or `main.py` file will be created, containing code that a user or explorer of the repo can simply run to visualize our results. This shall complete the project.

## Exploration guide
<a name="exploration-guide"></a>

As of this writing, I am currently in the process of [fine-tuning a RoBERTa model](https://colab.research.google.com/drive/1Md5Lpe4WiLxeDUzxZk8vdkRmA50MIUMh?usp=sharing) for my last stage of NER. If you wish to explore the project code, I suggest beginning with `CorpusProcessor_with_NER.py` in **[main files](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/main%20files)** (which actually does not contain any NER implementation within itself yet, despite being fully functional otherwise). It contains comprehensive data-preprocessing methods for tasks ranging from lemmatization to custom string-cleaning. If you run it as a script, it is meant to generate a processed object for `small_sample_text_for_testing.txt`(https://github.com/kkrishna24/deep_nlp_on_sf_literature/blob/main/Input%20Files/small_sample_text_for_testing.txt). I have not yet hosted my original corpus file on GitHub thanks to its size, but if you follow the documentation in `CorpusProcessor` itself, you can easily use your own corpus to test it out.

Most of the action as of the project's current version is contained in directories **[main files](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/main%20files)**, **[TF-IDF](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/TF-IDF)**, and **[LLM](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/LLM)**. In `main files`, after you've used `CorpusProcessor`, go to `NER.py` that performs first-stage NER on the corpus using [spaCy](https://github.com/explosion/spaCy) and stores the output in "NER_output2.csv" by default.

After performing basic NER, you can check out `sentences_of_entities.py`. If you run it as a script, it (attempts to) extract the famed 3.5M sentences from `sci-fi_text_cleaned.csv`, the csv file representing a [Pandas dataframe](https://github.com/pandas-dev/pandas) that contains the corpus as a list of sentences with their cleaned versions. Then from this list, `sentences_of_entities` extracts all the sentences that contain any of the entities that were found by spaCy's NER. (Note that these entities are brought in from `entity_list.txt` which is a file I created using a separate script, after running `NER.py`. `entity_list.txt` _is_ small enough to be included in the "main files" directory, so it is. Also note that in the current version of the code, "sci-fi_text_cleaned.csv" has not been added already to `main files`. You may have to create this file or slightly modify the code in `sentences_of_entities` to have it work as a script)

Now the results of `sentences_of_entities.py` will be stored in the `LLM` directory, in `sents_containing_named_entities.txt`. But don't go to the `LLM` directory just yet: first go to [`TF-IDF`](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/TF-IDF) and run `TF-IDF.py` as a script. This will use [scikit-learn](https://github.com/scikit-learn/scikit-learn)'s [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and KMeans clustering to extract a diverse batch of 8-10,000 sentences from `sents_containing_named_entities.txt`. The diverse batch of sentences will be stored in `representative_sents_for_GPT_labelling.txt` that remains in the `TF-IDF` directory. 
- The reason we use a representative sample, rather than the full 114,000+ entity-containing sentences is that otherwise, we'll need a _lot_ of [OpenAI API](https://github.com/openai/openai-python) tokens to run our round of custom NER using [GPT-3.5 Turbo](https://platform.openai.com/docs/guides/text-generation/chat-completions-api).

Finally, you can now go to directory [`LLM`](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/LLM), open `GPT_NER_round_1.py`, and run it as a script. I have already created data you can use, so when asked about it you can use any key except "Y" to generate RoBERTa training data. If you want to see real magic however, first go to `api_key.py` and insert a valid OpenAI API key. Then run `GPT_NER_round_1` again and say "Y" when it asks you for input. You can then see as we use [spaCyLLM](https://github.com/explosion/spacy-llm) and [OpenAI API](https://github.com/openai/openai-python) to label and extract terms from the text as `TECHNOLOGY`, `CONCEPT`, or `MISCELLANEOUS SIGNIFICANT`. Then, we [tokenize](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py) this output appropriately and prepare to use it for [fine-tuning a RoBERTa model](https://colab.research.google.com/drive/1Md5Lpe4WiLxeDUzxZk8vdkRmA50MIUMh?usp=sharing), so RoBERTa (documentation [here](https://huggingface.co/docs/transformers/model_doc/roberta)) can perform the remainder of our required NER - for free.

**Note on `LLM`**: `LLM_Interactor.py` is the module that handles the creation of - you guessed it - an LLM interaction object. It configures spaCyLLM - [please check the documentation](https://spacy.io/api/large-language-models) - using the configuration strings residing in `config_strings_2.py`, which in turn refers to `fewshot.json` for its [fewshot-training](https://blog.paperspace.com/few-shot-learning/) info. **This means that `LLM_Interactor` as well as `GPT_NER_round_1` are very independent and malleable modules.** By modifying `fewshot.json`, you can train GPT-3.5 to perform NER for any other kind of domain-specific task. You can also change the spacyLLM configuration by modifying `config_strings_2.py`, where you can modify the "labels" to be fed for your own domain-specific task. For example, you can change these lines:

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

**About the `LDA` modules**: In `main files`, note that we also have [`LDA.py`](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/main%20files/LDA.py) and [`LDATrainer.py`](https://github.com/kkrishna2023/deep_nlp_on_sf_literature/tree/main/main%20files/LDATrainer.py). These are in-themselves complete modules, which if you wish, you can repurpose, test or reuse. 
- To run `LDATrainer` as a script, change `FILEPATH` to "../Input Files/small_sample_text_for_testing.txt" or to a corpus of your own. The code-snippet (before you edit it) should look as below:

  <img width="800" alt="Screenshot 2024-01-25 at 11 57 27 AM" src="https://github.com/kkrishna24/deep_nlp_on_sf_literature/assets/121068842/bb3491f5-fa23-4f33-b185-c472df2fc7cb">

After I have completed NER with fine-tuned RoBERTa, I shall integrate LDA modeling and topic extraction into the project workflow.

**Note on `main.py`**: Prior to the final commit, a `main.py`/`app.py` file will be created, containing code to generate a visualization of our results on-demand. This shall complete the project. 

For more about me, see my [Linkedin profile](https://www.linkedin.com/in/krishnatripathi070/).

<!-- Below text, idk where to include yet but we'll include somewhere when LDA is done:

- After RoBERTa-powered NER, we shall be using `gensim` [LDA](https://radimrehurek.com/gensim/models/ldamodel.html) and `scikit-learn` to model topics, themes etc. for further analysis on this "substantial essence" (represented by tech, concept and character-containing sentences) of the corpus.
  - We are treating the sentences containing prominent domain-specific named-entities to be the essence of our corpus that we're concerned with. However, depending on the particular analysis _you_ want to perform, you might have extracted sentences containing other kinds of terms/entities as your essence. For example, if you wanted to model the _social_ realism of an SF corpus rather than its _scientific_ realism, or if you were examining the works of an author like [Octavia E. Butler](https://www.octaviabutler.com/theauthor), you might prefer to extract various gender-specific or worldbuilding-specific terms. Our corpus contains works more like [Isaac Asimov](https://www.britannica.com/biography/Isaac-Asimov)'s, which tend to be idea-heavy (i.e. tech and concept heavy). Thus it is reasonable to consider "tech" and "concept" terms are representing their essence, with "miscellaneous significant" capturing other theme or plot-important terms.
  - You might also model the "essence" of your corpus in another way altogether (for example, using sentiment analysis rather than NER). 
- We shall be using time-series analysis etc. to model the interplay of the concepts and tech-terms mentioned in the corpus. Our custom-NER output gives us a lot of leeway to examine the concepts that drive the SF corpus. Besides getting insights about how SF concepts change/evolve over the corpus, we might also get insights on how those concepts changed/evolved in actual real-world progression through the 20th century. (For example, if concepts do change over the corpus, that might simultaneously be a sign that the corpus itself represents stories that were written in a linear progression through time in the 20th century) This is likely because I am already aware that the corpus begins with the first issue of [the 'IF' magazine](https://gizmodo.com/the-entire-run-of-if-magazine-is-now-freely-available-o-1761691317) and several later texts are chronologically-later issues of the same magazine. However, after we're done with 'IF' magazine's issues, the text contains [Galaxy Magazine](https://www.theverge.com/2017/7/14/15970710/galaxy-science-fiction-magazine-online-free-reading-archive)'s early issues, which might reset our clock.
- A final `main.py` file will be created which will contain code to visualize our results. This shall complete the project. -->
