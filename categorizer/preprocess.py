#to clean and prepare text

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#ensure stopwords are downloaded (this is fine for local, can be part of Docker build)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

#rest of your preprocess_text function
def preprocess_text(text: str) -> str:
    #clean, remove stopwords, and stem the text.
    text = text.lower() 
    text = re.sub(r'\d+', '', text) 
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    words = text.split() 

    stop_words = set(stopwords.words('english')) 
    stemmer = PorterStemmer() 

    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned)