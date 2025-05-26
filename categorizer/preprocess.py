#to clean and prepare text

import re #Regular Expressions (to search, match, manipulate strings based on patterns)
import string
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer #words to root form

import nltk #Natural Language Toolkit (Python lib for NLP)
nltk.download('stopwords')

def preprocess_text(text: str) -> str:
    #clean, remove stopwords, and stem the text.
    text = text.lower() #to lowercase
    text = re.sub(r'\d+', '', text) #replace all digits with empty strings (coz nums don't really help classify website content)
    text = text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation characters
    words = text.split() #split text into words

    stop_words = set(stopwords.words('english')) #identify stopwords
    stemmer = PorterStemmer() #reduce words to their root form

    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned) #joined processed words into single string (separate by spaces)