from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Sample documents
documents = [
    "The cat chased the mouse.",
    "The mouse ran away."
]

document_file = "../Data Files (Readable)/corpus_as_cleaned_wordlist.txt"

with open(document_file) as file:
    books_as_strings = file.readlines()
    documents = books_as_strings[:50]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Generate the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms)
terms = vectorizer.get_feature_names_out()

scores = {
    "terms": [],
    "corresponding_scores": [],
}
# Print the TF-IDF scores for each term in each document
for i, doc in enumerate(documents):
    # print(f"Document {i+1}:")
    for j, term in enumerate(terms):
        tfidf_score = tfidf_matrix[i, j]
        scores["terms"].append(term)
        scores["corresponding_scores"].append(tfidf_score)
        # if tfidf_score > 0:
        #
        #   print(f"    {term}: {tfidf_score:.3f}")


with open("tf_idf_scores_for_first_50_books", "w") as file:
    json.dump(scores, file, indent=2)
