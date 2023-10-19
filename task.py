import random
random.seed(123)
import codecs
import string
from nltk.stem.porter import PorterStemmer
import gensim
import pickle


def createParagraphList(file):
    """
    file: string
    return: list(string)

    tar inn et filnavn, åpner fila, og returnerer en liste med en streng av hvert avsnitt
    """
    result = []
    with codecs.open(file, "r", "utf-8") as f:
        content = f.read()
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            p = p.strip()
            result.append(p)
    return result

def filtering(paragraphs, word):
    """
    paragraph: list(string)
    word: string
    return: list(string)
    
    Tar inn en liste med strenger, og fjerner de elementene som inneholder et gitt ord
    """
    result = []
    for p in paragraphs:
        if len(p) > 2:
            if word.lower() not in p.lower():
                    result.append(p)
    return result

def tokenize(paragraphs):
    """
    paragraphs: list(string)
    return: list(list(string))

    Tar inn en liste med avsnitt, deler hvert avsnitt opp i en liste med enkeltord, og returnerer alt
    """
    result = []
    for p in paragraphs:
        p_clean = ''.join(char for char in p if char not in (string.punctuation + '\n\r\t')).lower()
        words = p_clean.split(" ")
        result.append(words)
    return result

def stemmer(paragraphs):
    """
    paragraphs: list(list(string))
    return: list(list(string))

    Tar inn en liste med lister av strenger for hvert avsnitt, og stemmer hvert ord.
    """
    stemmer_instance = PorterStemmer()
    stemmedParagraphs = []
    for p in paragraphs:
        stemmedWords = []
        for w in p:
            stemmedWord = stemmer_instance.stem(w)
            stemmedWords.append(stemmedWord)
        stemmedParagraphs.append(stemmedWords)
    return stemmedParagraphs

def stopword(file):
    """
    file: string
    return: list(string)

    tar inn et filnavn, åpner fila, og returnerer en liste med en streng av hvert ord
    """
    result = []
    with codecs.open(file, "r", "utf-8") as f:
        content = f.read()
        stopwords = content.split(",")
        for s in stopwords:
            s = s.strip()
            result.append(s)
    return result

def preprocessing(query, dictionary):
    """
    query: string
    return: list(list(int, int))
    """
    queryList = [query]
    tokenizedQuery = tokenize(queryList)
    stemmedQuery = stemmer(tokenizedQuery)

    corpus = []
    for word in stemmedQuery: 
        corpus.append(dictionary.doc2bow(word))
    return corpus


paragraphs = createParagraphList("pg3300.txt")
filtered_paragraphs = filtering(paragraphs, "Gutenberg")


tokenized_paragraphs = tokenize(filtered_paragraphs)
with open('tokenized_paragraphs.pkl', 'wb') as file:
    pickle.dump(tokenized_paragraphs, file)
stemmed_paragraphs = stemmer(tokenized_paragraphs)
with open('stemmed_paragraphs.pkl', 'wb') as file:
    pickle.dump(stemmed_paragraphs, file)
"""
# Load tokenized data
with open('tokenized_paragraphs.pkl', 'rb') as file:
    tokenized_paragraphs = pickle.load(file)
# Load stemmed data
with open('stemmed_paragraphs.pkl', 'rb') as file:
    stemmed_paragraphs = pickle.load(file)
"""
dictionary = gensim.corpora.Dictionary(stemmed_paragraphs)
stopwords = stopword("stopwords.txt")
stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

corpus = []
for paragraph in stemmed_paragraphs: 
    corpus.append(dictionary.doc2bow(paragraph))

tfidf_model = gensim.models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]
tfidf_index = gensim.similarities.MatrixSimilarity(corpus, dictionary)

lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[tfidf_corpus]
lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus, num_features=len(dictionary))

#print(lsi_model.show_topics(3))

query = preprocessing("What is the function of money?", dictionary)
tfidf_query = tfidf_model[query][0]

print(len(paragraphs)," ", len(filtered_paragraphs)," ", len(tokenized_paragraphs)," ", len(stemmed_paragraphs))



