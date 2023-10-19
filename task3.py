import random
random.seed(123)
import codecs
import string
from nltk.stem.porter import PorterStemmer
import gensim

filename = "pg3300.txt"
file = codecs.open(filename, "r", "utf-8").read()

original_paragraphs = []

# Make an array with every paragraph in the text
for p in file.split("\n\n"):
    p = p.strip()  # Ensure the stripped paragraph is assigned back to 'p'
    original_paragraphs.append(p)

# Filter out all paragraphs containing "Gutenberg"
filtered_paragraphs = []
for p in original_paragraphs:
    if "Gutenberg" not in p:
        filtered_paragraphs.append(p)

# Tokenize paragraphs using .split()
paragraphs = [p.split() for p in filtered_paragraphs]

stemmer = PorterStemmer()

# Create a copy of the cleaned paragraphs
cleaned_paragraphs = []

# Go through every word in every paragraph, remove punctuation, convert to lowercase, and then apply stemmer
for words in paragraphs:
    cleaned_words = [stemmer.stem(''.join(char for char in word if char not in (string.punctuation + '\n\r\t')).lower()) for word in words]
    cleaned_paragraphs.append(cleaned_words)

dictionary = gensim.corpora.Dictionary(cleaned_paragraphs)

stopwords = codecs.open("stopwords.txt", "r", "utf-8").read().split(",")

# Find IDs of stopwords that exist in the dictionary
stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

corpus = []
for paragraph in cleaned_paragraphs: 
    corpus.append(dictionary.doc2bow(paragraph))

tfidf_model = gensim.models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]
matsim = gensim.similarities.MatrixSimilarity(corpus, dictionary)

lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[corpus]
lsi_sim = gensim.similarities.MatrixSimilarity(lsi_corpus)
lsi_model.show_topics()


query = ["What is the function of money?"]



print(1)