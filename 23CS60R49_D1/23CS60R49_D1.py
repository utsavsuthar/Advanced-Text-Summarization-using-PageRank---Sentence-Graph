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
for sentence in sentences:
    words = sentence.split()
    tf_dict.append(Counter(words))

#Calculate IDF (Inverse Document Frequency) for each word
word_sentence_count = Counter()
for sentence in sentences:
    words = set(sentence.lower().split())
    for word in words:
        word_sentence_count[word] += 1

total_sentences = len(sentences)

# Calculate IDF for each word
idf_dict = {}
for word, sentence_count in word_sentence_count.items():
    idf = math.log(total_sentences / (1 + sentence_count))
    idf_dict[word] = idf

unique_words = sorted(idf_dict.keys())

# Calculate TF-IDF for each word in each sentence and store in tfidf_matrix
tfidf_matrix = np.zeros((len(sentences), len(unique_words)))
for i, tf_sentence in enumerate(tf_dict):
    for j, word in enumerate(unique_words):
        tfidf_matrix[i, j] = tf_sentence[word] * idf_dict[word]

print("TF-IDF Matrix (NumPy array):")
tfidf_matrix=tfidf_matrix.T
print(tfidf_matrix)

#T3. Summarization — Using PageRank
cosine_similarities = cosine_similarity(tfidf_matrix.T)

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
for i in range(len(preprocessed_sentences)):
    if preprocessed_sentences[i] in summary_sentences:
        file.write(f"{sentences[i]}\n")

# Closing file
file.close()
