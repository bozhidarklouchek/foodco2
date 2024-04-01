from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_most_relevant_sentence_index(term, sentences):
    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    index_vote = {}

    # Case fold & break down term
    term = term.lower()
    for part in term.split(' '):
        part_index = vectorizer.vocabulary_.get(part)
        if part_index is None:
            continue

        # Get TF-IDF scores for the given term across all sentences and get max index
        term_tfidf_scores = X[:, part_index]
        most_relevant_index = np.argmax(term_tfidf_scores)
        
        if most_relevant_index in index_vote.keys():
            index_vote[most_relevant_index] += 1
        else:
            index_vote[most_relevant_index] = 1

    if(len(index_vote.keys())) == 0:
        return -1

    # Return the most relevant sentence by vote of parts
    max_key = max(index_vote, key=index_vote.get)
    return max_key

# # Example usage
# term = "lean ground beef"
# sentences = [
#     "Ground beef is good for you!",
#     "The mix the beef with the peppers.",
#     "Eating raw beef from the ground can be really bad for you!"
# ]

# # Example usage
# term = "lean ground beef"
# sentences = [
#     "ground",
#     "ground",
#     "woeiuf iewu jmfw "
# ]


# most_relevant_index = get_most_relevant_sentence_index(term.lower(), sentences)
# print("Most relevant sentence:", sentences[most_relevant_index])
