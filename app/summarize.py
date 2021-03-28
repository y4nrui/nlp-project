
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from summarizer import Summarizer


model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, world!'

@app.route('/summarize', methods=['POST']) 
def summarize():  
    text = request.form.get('text') # text that the user input
    models = request.form.get('model') # summarization model that user chooses
    print("successfully received", text, "and", models)
    

    if models == 't5':
        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        # summmarize 
        summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=10000,
                                    early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        text=output
        return jsonify({"result": text, "model": models}) # returns a json of text
    
    elif models == 'bert':
        model2 = Summarizer()
        text = model2(text)
    
        return jsonify({"result": text, "model": models}) # returns a json of text

    
    elif models == 'pagerank':
        sentences=(sent_tokenize(text))
        word_embeddings = {}
        f = open('./word_embeddings/glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()
        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]
        stop_words = stopwords.words('english')
        # function to remove stopwords
        def remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in stop_words])
            return sen_new
        # remove stopwords from the sentences
        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
        # Extract word vectors
        word_embeddings = {}
        # f = open('C:\\wamp64\\www\\nlp-project-main\\data\\glove.6B.100d.txt', encoding='utf-8')
        f = open('./word_embeddings/glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()
        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)
        # similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        # Extract top 10 sentences as the summary
        for i in range(1):
            text=(ranked_sentences[i][1])
            return jsonify({"result": text, "model": models}) # returns a json of text
        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)  # Enable reloader and debugger