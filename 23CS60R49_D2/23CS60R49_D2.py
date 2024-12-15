import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# T1. Preprocessing and obtaining sentences using NLTK:
sentences=[]
with open('input.txt') as f:
    for line in f:
        sentences.append(line.replace('\n', '').strip())

for sentence in enumerate(sentences):
    print(sentence)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

preprocessed_sentences = []

for sentence in sentences:
    # Remove punctuation and convert to lowercase
    translator = str.maketrans("", "", string.punctuation)
    sentence = sentence.translate(translator).lower()
    # sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Remove stopwords and lemmatize words
    words_without_stopwords=[]
    for word in words:
        if word not in stop_words:
            words_without_stopwords.append(word)
    lemmatized_words=[]
    for word,tag in pos_tag(words_without_stopwords):
        if tag.startswith('J'):
            pos = 'a' #Adjective
        elif tag.startswith('V'):
            pos = 'v' #Verb
        elif tag.startswith('N'):
            pos = 'n' #Noun
        elif tag.startswith('R'):
            pos = 'r' #Adverb
        else:
            pos='n'
        lemmatized_word=lemmatizer.lemmatize(word,pos)
        lemmatized_words.append(lemmatized_word)
    
    preprocessed_sentence = ' '.join(lemmatized_words)
    preprocessed_sentences.append(preprocessed_sentence)


for i, sentence in enumerate(preprocessed_sentences):
    print(f"List[{i}] = {sentence}")

#T2. Sentence Representation:
# Calculate TF (Term Frequency) for each word in each sentence
tf_dict = []
for sentence in preprocessed_sentences:
    words = sentence.split()
    tf_dict.append(Counter(words))

#Calculate IDF (Inverse Document Frequency) for each word
word_sentence_count = Counter()
for sentence in preprocessed_sentences:
    words = set(sentence.lower().split())
    for word in words:
        word_sentence_count[word] += 1

total_sentences = len(preprocessed_sentences)

# Calculate IDF for each word
idf_dict = {}
for word, sentence_count in word_sentence_count.items():
    idf = math.log(total_sentences / (1 + sentence_count))
    idf_dict[word] = idf

unique_words = sorted(idf_dict.keys())

# Calculate TF-IDF for each word in each sentence and store in tfidf_matrix
tfidf_matrix = np.zeros((len(preprocessed_sentences), len(unique_words)))
for i, tf_sentence in enumerate(tf_dict):
    for j, word in enumerate(unique_words):
        tfidf_matrix[i, j] = tf_sentence[word] * idf_dict[word]

print("TF-IDF Matrix (NumPy array):")
tfidf_matrix=tfidf_matrix
# print(tfidf_matrix)

#T3. Summarization — Using PageRank
cosine_similarities = cosine_similarity(tfidf_matrix)

# Create a graph
graph = nx.Graph()

# Add vertices (nodes)
for i in range(total_sentences):
    graph.add_node(i, sentence=preprocessed_sentences[i])  # Add sentence as an attribute
    
# Add edges with weights (cosine similarities)
for i in range(total_sentences):
    for j in range(i + 1, total_sentences):  # Only consider upper triangle to avoid duplicates
        similarity = cosine_similarities[i, j]
        graph.add_edge(i, j, weight=similarity)
edge_weights = {(u,v): d["weight"] for u,v,d in graph.edges(data=True)}
# print(len(edge_weights))
pos= nx.spring_layout(graph)
nx.draw(graph,pos,with_labels=True, node_size=1000)
nx.draw_networkx_edge_labels(graph,pos,edge_labels=edge_weights)
plt.show()
# Display the edges and weights
# for edge in graph.edges(data=True):
#     print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]['weight']:.4f}")



# Compute PageRank
pagerank_scores = nx.pagerank(graph)

# Convert pagerank_scores dictionary to a list of tuples (node_id, pagerank)
pagerank_list = [(node_id, score) for node_id, score in pagerank_scores.items()]

# Sort the list by PageRank values in descending order
pagerank_list.sort(key=lambda x: x[1], reverse=True)
print(pagerank_list)

top_n = 4
summary_sentences = [preprocessed_sentences[node_id] for node_id, _ in pagerank_list[:top_n]]

# write output in file Summary_PR.txt
# Opening a file
file = open('Summary_PR.txt', 'w')

# Writing a string to file
# Display the summary sentences in the order they appear in the input document
# for i in range(len(preprocessed_sentences)):
#     if preprocessed_sentences[i] in summary_sentences:
#         file.write(f"{sentences[i]}\n")

# Closing file
# file.close()

Selected_sentences=[]
with open('Summary_PR.txt') as f:
    for line in f:
        Selected_sentences.append(line.replace('\n', '').strip())

for sentence in enumerate(Selected_sentences):
    print(sentence)

not_selected_sentences=[sentence for sentence in sentences if sentence not in Selected_sentences]
print(not_selected_sentences)

# Initial ranked sentences (top-3)
top_n = 3
initial_ranking = []
for i in range(top_n):
    initial_ranking.append(pagerank_list[i][0])

lambda_param = 0.5  
summary_length = 5 
summary = initial_ranking[:]

# Rerank using MMR
while len(summary) < summary_length:
    best_candidate = None
    best_mmr_score = -np.inf

    for candidate in range(len(preprocessed_sentences)):
        if candidate not in summary:
            IMP = pagerank_scores[candidate]
            Max_Sim = max([cosine_similarities[candidate][selected] for selected in summary])
            mmr_score = lambda_param * IMP - (1 - lambda_param) * Max_Sim

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = candidate

    if best_candidate is not None:
        summary.append(best_candidate)

reranked_summary = [sentences[idx] for idx in summary]

for i in range(len(reranked_summary)):
    file.write(f"{reranked_summary[i]}\n")

cosine_similarities = cosine_similarity(tfidf_matrix)
K=3
index=[]
for i in range(total_sentences):
    index.append(i)

import random
Random_Centroids = (random.sample(index, 3))
centroids = tfidf_matrix[Random_Centroids]

max_iterations = 100  
for _ in range(max_iterations):
    cosine_similarities = cosine_similarity(tfidf_matrix, centroids)
    cluster_assignments = np.argmax(cosine_similarities, axis=1)
    new_centroids = np.array([tfidf_matrix[cluster_assignments == i].mean(axis=0) for i in range(K)])
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

clusters = {}
for i in range(K):
    clusters[i] = [sentences[j] for j in range(len(sentences)) if cluster_assignments[j] == i]

for cluster_id, cluster_sentences in clusters.items():
    print(f"Cluster {cluster_id + 1}:")
    for sentence in cluster_sentences:
        print(sentence)
    print()

closest_sentences = {}
cosine_similarities = cosine_similarity(tfidf_matrix, centroids)

for cluster_id in range(K):
    cluster_indices = np.where(cluster_assignments == cluster_id)[0]
    mean_similarity_scores = cosine_similarities[cluster_indices].mean(axis=1)
    closest_sentence_index = cluster_indices[np.argmax(mean_similarity_scores)]
    closest_sentences[cluster_id] = sentences[closest_sentence_index]

for cluster_id, closest_sentence in closest_sentences.items():
    print(f"Cluster {cluster_id + 1}:")
    print("Closest Sentence:", closest_sentence)
    print()

from nltk.util import ngrams
from collections import Counter

def common_bigrams(sentence1, sentence2, n=2):
    bigrams1 = set(ngrams(sentence1.split(), n))
    bigrams2 = set(ngrams(sentence2.split(), n))
    common = bigrams1.intersection(bigrams2)
    return common

common_bigram_sentences = {}

# Iterate through each cluster
for cluster_id in range(K):
    cluster_sentences = [sentences[i] for i in range(len(sentences)) if cluster_assignments[i] == cluster_id]
    
    # Find S1 and S2
    s1 = closest_sentences[cluster_id]
    s2 = None
    

    for sentence in cluster_sentences:
        common_bigrams_count = len(common_bigrams(s1, sentence))
        if common_bigrams_count >= 3:
            s2 = sentence
            break
    
  
    if s2:
        common_bigram_sentences[cluster_id] = (s1, s2)
    else:
        common_bigram_sentences[cluster_id] = (s1,)


for cluster_id, sentences_tuple in common_bigram_sentences.items():
    s1, s2 = sentences_tuple
    print(f"Cluster {cluster_id + 1}:")
    print("S1:", s1)
    if s2:
        print("S2:", s2)
    print()
