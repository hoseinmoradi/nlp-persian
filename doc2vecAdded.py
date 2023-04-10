# coding: utf8
import gensim
import gensim.models.doc2vec
from gensim.summarization import summarize
from IPython.display import display
from json import JSONEncoder
from collections import defaultdict
from flask_cors import CORS, cross_origin
from flask import jsonify
import requests
import urllib.request
from flask import Flask, request
from keywords import keywords
from commons import build_graph
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from hazm import *
import collections
from wordcloud_fa import WordCloudFa
import numpy as np

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/GetSummarize', methods=['POST'])
def GetSummarize():
    text = request.form.get('text')
    indx = { "data" : summarize(text)}
    return jsonify(indx)


@app.route('/GetKeyword',methods=['POST'])
def GetKeywords():
    request_key = request.form.get('text')
    indx = { "data" : keywords(request_key, ratio=0.1, words=None, split=False, scores=True, pos_filter=('NN', 'JJ'),
             lemmatize=True, deacc=False)}
    return jsonify(indx)

@app.route('/wordcloud',methods=['POST'])
def getWordCloud():
    text = request.form.get('text')
    normalizer = Normalizer()
    stemmer = Stemmer()
    text = normalizer.normalize(text)
    words = word_tokenize(text)
    stopwords = stopwords_list()
    words = [word for word in words if word not in stopwords]
    stemmed_words = [stemmer.stem(word) for word in words]
    word_count = collections.Counter(stemmed_words)
    keywords = dict(word_count.most_common(30))
    wordcloud = WordCloudFa(persian_normalize=True)
    wc = wordcloud.generate_from_frequencies(keywords)
    image = wc.to_image()
    image.show()
    image.save('wordcloud.png')
    
    
@app.route('/informationExtraction',methods=['POST'])
def informationExtraction():
    text = request.form.get('text')
    normalizer = Normalizer()
    text = normalizer.normalize(text)
    words = word_tokenize(text)
    stemmer = Stemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    # Tagging the words
    tagger = POSTagger(model='resources/postagger.model')
    tags = tagger.tag(words)

    # Extracting named entities
    chunker = Chunker(model='resources/chunker.model')
    tree2brackets = TreebankReader().tree2brackets
    brackets2str = BracketParseCorpusReader().bracket2str
    chunks = chunker.chunk(tagged_sentence=tags)
    brackets = tree2brackets(chunks)
    entities = brackets2str(brackets)

    # Displaying the extracted information
    # print("Words: ", words)
    # print("Stemmed words: ", stemmed_words)
    # print("POS tags: ", tags)
    # print("Named entities: ", entities)
    return jsonify(entities)


# Output:


# Words:  ['من', 'به', 'دانشگاه', 'تهران', 'رفتم', 'و', 'در', 'رشته', 'مهندسی', 'کامپیوتر', 'تحصیل', 'کردم', '.']
# Stemmed words:  ['من', 'به', 'دانشگاه', 'تهران', 'رفت', 'و', 'در', 'رشته', 'مهندسی', 'کامپیوتر', 'تحصیل', 'کرد', '.']
# POS tags:  [('من', 'PRO'), ('به', 'P'), ('دانشگاه', 'NC'), ('تهران', 'NP'), ('رفتم', 'V'), ('و', 'CONJ'), ('در', 'P'), ('رشته', 'NC'), ('مهندسی', 'NC'), ('کامپیوتر', 'NC'), ('تحصیل', 'NC'), ('کردم', 'V'), ('.', '.')]
# Named entities:  (PP دانشگاه تهران/NP) (PP رشته مهندسی کامپیوتر/NC)

#documents = gensim.models.doc2vec.TaggedLineDocument('result2.txt')
#loading the model
#model_loaded = gensim.models.doc2vec.Doc2Vec.load('my_model_sents_from_res2.doc2vec')

#sample Text
if __name__ == '__main__':
    app.debug = True
    app.run()