# Classes to perform initial TF-IDF for extracting diverse batches of sentences, and perform TF-IDF
# for topic modeling later

# Benefits: Many, but also most prominently, well-eliminates the issue of irrelevant blocks of text
#           composed of editorial discussions etc.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


class BatchMaker:
    def __init__(self, all_sentences, num_clusters=10, num_batches=4, batch_size=2000,
                 maintain_text_integrity=True):
        """Input a list of sentences and get a diverse representation which can be accessed later
        through the *representative_sentences* attribute.

        *num_clusters* is the number of k-means clusters to make using sklearn.cluster. 10 is default,
        and reasonable.

        *num_batches* is how many batches of sentences you want from different portions of the text.

        *batch_size* is the size of each batch. The total no. of representative sentences you get is
        equal to num_batches * batch_size.

        *maintain_text_integrity* tells the maker whether to remove stop words or not. By default,
        it does not."""

        self.representative_sentences = []  # our main result-attribute

        self.all_sentences = all_sentences  # what we have to deal with

        # Setting attributes to be filled in by our methods later
        self.matrix, self.cluster_assignments, self.batches = None, None, None

        # Initialization methods
        self.vectorize(self.all_sentences, maintain_text_integrity)
        self.create_clusters(self.matrix, num_clusters)

        # Final step
        self.create_batches(num_clusters, num_batches, batch_size)

    def vectorize(self, sentences, maintain_text_integrity):
        # Convert sentences to a TF-IDF matrix
        if maintain_text_integrity:
            vectorizer = TfidfVectorizer()
        else:
            vectorizer = TfidfVectorizer(stop_words='english')

        x = vectorizer.fit_transform(sentences)
        self.matrix = x

        return x  # returning also just in case someone wants to use for an independent purpose

    def create_clusters(self, document_term_matrix, n_clusters, random_seed=42):
        # Set up KMeans with the input number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)

        # Use to compute the KMeans clustering for the input matrix
        kmeans.fit(document_term_matrix)

        # Assign each sentence to a cluster
        self.cluster_assignments = kmeans.predict(document_term_matrix)

    def create_batches(self, num_clusters, num_batches, batch_size):
        cluster_assignments = self.cluster_assignments  # NOTE: The create_clusters method MUST have run before this
        all_sentences = self.all_sentences

        # Randomly sample sentences from each cluster to create diverse batches
        batch_size = batch_size
        batches = []

        for _ in range(num_batches):
            selected_sentences = []
            for cluster_id in range(num_clusters):
                cluster_indices = np.where(cluster_assignments == cluster_id)[0]

                # This line would select sentences randomly from the cluster
                # cluster_samples = np.random.choice(cluster_indices, size=batch_size // num_clusters, replace=False)

                # Select sentences sequentially from each cluster
                start_index = (_ * batch_size // num_clusters) % len(cluster_indices)
                end_index = start_index + (batch_size // num_clusters)
                cluster_samples = cluster_indices[start_index:end_index]

                selected_sentences.extend([all_sentences[i] for i in cluster_samples.tolist()])

            batches.append(selected_sentences)

        # Now 'self.batches' and 'batches' both will contain num_batches number of lists with batch_size sentences
        self.batches = batches

        # Fill the 'representative_sentences' attribute of the class object with a contiguous list of these sentences
        for i in range(num_batches):
            self.representative_sentences.extend(batches[i])


if __name__ == '__main__':

    with open("../LLM/sents_containing_named_entities.txt", "r") as file:

        # Read the whole file as one string
        as_string = file.read()

        # Now split the file into sentences based on newlines
        sentences = as_string.splitlines()

        # Check
        print("Verify inputs below")
        print("Check number of sentences:", len(sentences))
        print("First few:", sentences[:30])

    all_sents = sentences

    # Make Batchmaker object with default settings
    batchmaker = BatchMaker(all_sents)

    # Check out the representative sentences
    rep_sents = batchmaker.representative_sentences

    print("num_extracted_sents:", len(rep_sents))

    # Check if the sample sentences are in order and diversity like we want it to me
    for sent in rep_sents[:200]:  # first 200 sentences
        print(all_sents.index(sent))

    # Visually check the batch quality. Beautiful
    print("Batches:\n")
    for batch in batchmaker.batches:  # Assuming that each batch is sized 2000!
        print("First 30:", batch[:30])
        print("Mid 30:", batch[1000:1031])
        print("Last 30:", batch[1970:])

    # Write the representative sentences to a file to be used for NER with spacy-llm
    with open("representative_sents_for_GPT_labelling.txt", "w") as outfile:
        # Write each sentence from the list to a new line in the file
        for sent in rep_sents:
            outfile.write("%s\n" % sent)

