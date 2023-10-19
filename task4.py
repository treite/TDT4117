import random
random.seed(123)
import codecs
import string
from nltk.stem.porter import PorterStemmer
import gensim

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

paragraphs = createParagraphList("pg3300.txt")
filtered_paragraphs = filtering(paragraphs, "Gutenberg")
tokenized_paragraphs = tokenize(filtered_paragraphs)
stemmed_paragraphs = stemmer(tokenized_paragraphs)

print(len(paragraphs)," ", len(filtered_paragraphs)," ", len(tokenized_paragraphs)," ", len(stemmed_paragraphs))


print(1)


