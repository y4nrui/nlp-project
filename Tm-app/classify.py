from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

import pickle

us_to_uk_mapping = pickle.load(open("dep/us_gb_updated.pkl", "rb"))

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")

import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)

import numpy as np
import pandas as pd
import nltk
import re

app = Flask(__name__)
CORS(app)

expletives_mapping_df = pd.read_csv('dep/expletives-corrected.csv', index_col=None)
stopwords_to_be_excluded = ['US'] # list of words for spacy to keep if they're a stopword
key_acronyms_mapping = {"u.s.": "US", "l.a.": "LA", "w.h.": "WH", "g20": "gtwenty",
                        "g-20": "gtwenty", "d.c.": "DC", "u.k.": "UK", "n.c.": "NC", "s.c.": "SC"} 

# Importing NN Dependencies
tfidf_vec = pickle.load(open("dep/unigram_tfidf_vectorizer.pk", "rb"))
nbc = pickle.load(open("dep/unigram_nb_model.sav", "rb"))
rfc = pickle.load(open("dep/unigram_rf_model_2.sav", "rb"))

base_tok = pickle.load(open("dep/base_lstm_tokenizer.pk", "rb"))
uni_dir_lstm_model = tf.keras.models.load_model('dep/uni_dir_lstm_model')
bi_dir_lstm_model = tf.keras.models.load_model('dep/bi_dir_lstm_model')

glove_tok = pickle.load(open("dep/glove_lstm_tokenizer.pk", "rb"))
glove_lstm_model = tf.keras.models.load_model('dep/glove_lstm_model')

w2v_tok = pickle.load(open("dep/w2v_tokenizer.pk", "rb"))
w2v_lstm_model = tf.keras.models.load_model('dep/w2v_lstm_model')


def text_preprocessor(x):
    '''
    Parameters
    ----------
    x : str
        The string to be cleaned
        
    Using pure spaCy instead of NLTK for faster preprocessing. Function will return None if there's an error in the cleaning.

    Preprocessing steps:
    ---------------------
    1. Check if <type> param is filled correctly
    2. Removal of source of article at the start of each text (for true news only)
    3. Removal of whitespace
    4. Convert to lowercase
    5. Handling key punctuated acronyms (eg. "U.S.") 
    6. Removal of words within brackets
    7. Removal of URLs eg. "bit.ly/..." or "pic.twitter.com/..."
    8. Removal of punctuations and special characters and numbers, except asterisks
    9. Uncensoring expletives
    10. Removal of stop words, tokenisation and lemmatization, handling of us spelling words, and characters not cleaned previously

    '''

    try: 
        # Removal of leading and trailing whitespace characters eg. '\n', '\r' 
        x = x.strip()

        # Convert to lowercase
        x = x.lower()

        # Replace important punctuated key acronyms with non-punctuated versions
        for key, value in key_acronyms_mapping.items():
            x = x.replace(key, value)

        # Removal of words within brackets
        x = re.sub(r"[\(\[].*?[\)\]]", "", x)
        
        # Removal of URLs
        x = re.sub(r'https?:\S+', '', x) # remove URLs that contain http or https
        x = re.sub(r'pic.twitter.com/[\w]*', '', x) # remove twitter pic URLs
        x = re.sub(r'tmsnrt.rs/[\w]*', '', x) # remove tmnsrt.rs URLs (thomson reuters)
        x = re.sub(r'reut.rs/[\w]*', '', x) # remove reut.rs URLs (reuters)
        x = re.sub(r'bit.ly/[\w]*', '', x) # remove bit.ly URLs
        x = re.sub(r'[\w]*.com/\S+', '', x) # remove URLs generally ie. words that contain ".com/"
        

        # Removal of punctuations and special characters, except asterisk (*) characters because of words like "f**k", replaces the unwanted characters with an empty whitespace.
        x = re.sub(r'[^\w\s\*]',' ',x) 

        # Removal of numbers
        x = re.sub(r'\d+', '', x)
        
        # Uncensoring expletives
        for i in range(len(expletives_mapping_df)):
            x = x.replace(expletives_mapping_df['expletives'][i], expletives_mapping_df['corrected'][i])


        # Removal of stop words, tokenisation and lemmatization, handling of us spelling words, and characters not cleaned previously
        doc = nlp(x) # initialising spaCy object
        result = []
        for token in doc: # iterate through spaCy object: for each token in the text
            if len(token.text.strip()) != 0: # prevent empty whitespace (caused by punctuations) from being stored in the result
                if len(token.text) > 1: # exclude asterisk and other characters that won't be meaningful
                    if token.is_stop == False or token.text in stopwords_to_be_excluded: # if token.text is not stopwords, or if token.text is a stopword that we want to keep
                            lem_token = token.lemma_
                            if lem_token in us_to_uk_mapping: #   if token.text is a US spelling word, convert to UK
                                result.append(us_to_uk_mapping[lem_token]) # lemmatize and add into np.array
                            else:
                                result.append(lem_token)
        result_str = (" ".join(result)).lower()
        return result_str

    except: # if any error happens, the preprocessor will just return None instead of an error that kills the preprocessing code runtime
        return None



@app.route('/')
def hello():
    return 'Hello, world!'

@app.route('/classify', methods=['POST']) 
def classify():  
    text = request.form.get('text') # text that the user input
    models = request.form.get('model') # summarization model that user chooses
    print("successfully received text and models")
    
    cleaned_text = text_preprocessor(text)
    print(cleaned_text)

    if cleaned_text == None:
        return jsonify({"result": None}) # returns a json of text

    elif models == 'nbc':

        
        vector = tfidf_vec.transform([cleaned_text])
        prediction = nbc.predict(vector)[0]
        print(prediction)
        return jsonify({"result": str(prediction)})

    elif models == 'rfc':

        
        vector = tfidf_vec.transform([cleaned_text])
        prediction = rfc.predict(vector)[0]
        print(prediction)
        return jsonify({"result": str(prediction)})
    
    # uni_lstm, bi_lstm, glove_lstm, w2v_lstm

    elif models == "uni_lstm":

        vector = base_tok.texts_to_sequences([cleaned_text])
        vector = tf.keras.preprocessing.sequence.pad_sequences(vector, padding='post', maxlen=256)
        prediction = uni_dir_lstm_model.predict(vector)[0][0]
        print(prediction)
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
        
        return jsonify({"result": str(prediction)})

    elif models == "bi_lstm":

        vector = base_tok.texts_to_sequences([cleaned_text])
        vector = tf.keras.preprocessing.sequence.pad_sequences(vector, padding='post', maxlen=256)
        prediction = bi_dir_lstm_model.predict(vector)[0][0]
        print(prediction)
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
        
        return jsonify({"result": str(prediction)})

    elif models == "glove_lstm":

        vector = glove_tok.texts_to_sequences([cleaned_text])
        vector = tf.keras.preprocessing.sequence.pad_sequences(vector, maxlen=250)
        prediction = glove_lstm_model.predict(vector)[0][0]
        print(prediction)
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
        
        return jsonify({"result": str(prediction)})

    elif models == "w2v_lstm":

        vector = w2v_tok.texts_to_sequences([cleaned_text])
        vector = tf.keras.preprocessing.sequence.pad_sequences(vector, maxlen=500)
        prediction = w2v_lstm_model.predict(vector)[0][0]
        print(prediction)
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
        
        return jsonify({"result": str(prediction)})
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002,debug=True)  # Enable reloader and debugger