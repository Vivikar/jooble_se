#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
"""This service contains forward_index, inverted_index and list of documents_id.

Structure of forward_index:
    dict {"id_doc1": Document instance1,
          "id_doc2": Document instance2,
          ...}
    
Structure of inverted_index:
    dict {"term1": ["doc_id1", "doc_id2", ...],
          "term2": ["doc_id1", "doc_id2", ...],
          ...}
          
Structure of documents_id:
    list of str ["id_doc1", "id_doc2", ...]

"""

import json
import os
import pickle

from flask import Flask, request
import jsonpickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from config_src import config
from document import Document

#doc to vec from gensim
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


#%%
forward_index = {}
inverted_index = {}
documents_id = []
vectorizer_tfidf = None


#%%

def save_index(path, forward_index, inverted_index, documents_id,
               forward_file="forward_index", 
               inverted_file="inverted_index", 
               id_file="documents_id"):
    
    """Save index as json files
    
    :param str path: path to folder
    :param dict forward_index: link to forward_index instance.
    :param dict inverted_index: link to inverted_index instance.
    :param list of str documents_id: link to documents_id  instance.
    
    :param str forward_file: file name for forward_index without extension.
    :param str inverted_file: file name for inverted_index without extension.
    :param str id_file: file name for documents_id without extension.
    """
    
    file_path = os.path.join(path, forward_file + ".json")
    with open(file_path, 'w', encoding='utf8') as outfile:
        forward_index = jsonpickle.encode(forward_index)
        json.dump(forward_index, outfile, ensure_ascii=False)

    file_path = os.path.join(path, inverted_file + ".json")
    with open(file_path, 'w', encoding='utf8') as outfile:
        inverted_index = jsonpickle.encode(inverted_index)
        json.dump(inverted_index, outfile, ensure_ascii=False)

    file_path = os.path.join(path, id_file + ".json")
    with open(file_path, 'w') as outfile:
        documents_id = jsonpickle.encode(documents_id)
        json.dump(documents_id, outfile)

        
def load_index(path, forward_file="forward_index", 
               inverted_file="inverted_index", 
               id_file="documents_id"):
    """Load index from files.
    
    If files don't exist, return empty entities.
    
    :return dict forward_index:
    :return dict inverted_index:
    :return list of int documents_id:
    """
    forward_index = {}
    inverted_index = {}
    documents_id = []
    
    file_path_forw = os.path.join(path, forward_file + ".json")
    file_path_inv = os.path.join(path, inverted_file + ".json")
    file_path_id = os.path.join(path, id_file + ".json")
    
    files_present = os.path.exists(file_path_forw) and                     os.path.exists(file_path_inv) and                     os.path.exists(file_path_id)
    
    if files_present:    
        with open(file_path_forw, 'r', encoding='utf8') as infile:
            forward_index = json.load(infile)
            forward_index = jsonpickle.decode(forward_index)
    
        with open(file_path_inv, 'r', encoding='utf8') as infile:
            inverted_index = json.load(infile)
            inverted_index = jsonpickle.decode(inverted_index)
    
        with open(file_path_id, 'r', encoding='utf8') as infile:
            documents_id = json.load(infile)
            documents_id = jsonpickle.decode(documents_id)
            
    return forward_index, inverted_index, documents_id


def search_boolean(search_query, forward_index, inverted_index, documents_id):
    """Search the words of search_query in inverted index
    
    :param str search_query: normilized with stemming
    :param dict forward_index:
    :param dict inverted_index:
    :param list of str documents_id:
    
    :return list of Document documents: returns empty list if documents aren't found
    """
    docs_id = []
    words_in_query = search_query.split(" ")
    words = []
    for word in words_in_query:
        if word in inverted_index.keys():
            docs_id.append(set(inverted_index[word]))
            words.append(word)                         
    documents = []
    if len(docs_id) > 0:
        set_of_docs_id = docs_id[0]
        for docs_set in docs_id:
            set_of_docs_id = set_of_docs_id.intersection(docs_set)
                           
        for id in set_of_docs_id:
            documents.append(forward_index[str(id)])                 
    return documents


def add_forward_index(document, forward_index):
    """Add the document to forward index.
    
    :param Document document:
    :param dict forward_index:
    
    :return dict forward_index: Updated with new document.
    """
    forward_index[str(document.id)] = document
    return forward_index


def add_inverted_index(document, inverted_index):
    """Add the document to inverted index.
    
    :param Document document: document must be preprocessed 
    :param dict inverted_index:
    
    :return dict inverted_index: Updated with new document.
    """
    tokens = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(document)]
    for token in tokens:
        if token in inverted_index.keys():
            inverted_index[token].append(document.id)
        else:
            inverted_index[token] = [document.id]
    return inverted_index

def load_vectorizer(path, file="vectorizer.dat"):
    """
    Load tfidf vectorizer from file.
    
    If files don't exist, form vectorizer from index.

    :return TfidfVectorizet vectorizer:
    """ 
    file_path = os.path.join(path, file)
    
    file_present = os.path.exists(file_path)
  
    if file_present:    
        with open(file_path, "rb") as inf:
            vectorizer = pickle.load(inf) 
    else:
        vectorizer = build_tfidf_from_index(file_path)
    return vectorizer


def build_from_index(save_tfidf_path): 
    """
    :param str save_tfidf_path:
    :retunt TfidfVectorizer vectorizer:
    """
    corpus = []
    for i in documents_id:
        corpus.append(forward_index[i].text_normalized)
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    
    save_tfidf_path = os.path.join(save_tfidf_path, "vectorizer_tfidf.dat")
    with open(save_tfidf_path, "wb") as ouf:
        pickle.dump(vectorizer, ouf)          
    return vectorizer


def ranking(documents, query, vectorizer):
    """
    Rank documents by cosine similarity
    
    :param list of Document documents:
    :param str query: must be normilized with stemming
    :param TfidfVectorizer vectorizer:
    :return list of Document ranked_list:
    """
    query_vect = vectorizer.transform([query]).todense()
    doc_vects = [doc.text_normalized for doc in documents]
    doc_vects = vectorizer.transform(doc_vects).todense()
    
    ranked_list = cosine_similarity(doc_vects, query_vect)
    ranked_list = list(np.squeeze(ranked_list, axis=1))
    assert(len(ranked_list) == len(documents))
    
    # take into attention title
    doc_vects = [str(doc.title_normalized) for doc in documents]
    doc_vects = vectorizer.transform(doc_vects).todense()
    
    ranked_list_title = cosine_similarity(doc_vects, query_vect)
    ranked_list_title = list(np.squeeze(ranked_list_title, axis=1))
    assert(len(ranked_list_title) == len(documents))
    
    ranked_list = 0.5 * np.array(ranked_list) + 0.5 * np.array(ranked_list_title)
    ranked_list = list(ranked_list)
    
    ranked_list = list(zip(ranked_list, documents))
    ranked_list = sorted(ranked_list, key=lambda x: x[0]) 
    ranked_list = list(reversed(ranked_list))
    ranked_list = [i[1] for i in ranked_list]
    return ranked_list


def nn_rank_with_requirments(documents, query, vectorizer):
    query_vect = vectorizer.transform([query]).todense()
    
    good_docs = [doc for doc in documents if len(str(doc.requirement_normalized)) > 0]
    bad_docs = [doc for doc in documents if len(str(doc.requirement_normalized)) == 0]
    #print(good_docs)
    
    ranked_list = []
    if len(good_docs) > 0:
        doc_vects = [doc.text_normalized for doc in good_docs]
        doc_vects = vectorizer.transform(doc_vects).todense()
        
        ranked_list = cosine_similarity(doc_vects, query_vect)
        ranked_list = list(np.squeeze(ranked_list, axis=1))
        assert(len(ranked_list) == len(good_docs))
    
        ranked_list = list(zip(ranked_list, good_docs))
        ranked_list = sorted(ranked_list, key=lambda x: x[0]) 
        ranked_list = list(reversed(ranked_list))
     
    ranked_list2 = []
    if len(bad_docs) > 0:
        #bad docs
        doc_vects = [doc.text_normalized for doc in bad_docs]
        doc_vects = vectorizer.transform(doc_vects).todense()
        
        ranked_list2 = cosine_similarity(doc_vects, query_vect)
        ranked_list2 = list(np.squeeze(ranked_list2, axis=1))
        assert(len(ranked_list) == len(bad_docs))
    
        ranked_list2 = list(zip(ranked_list2, bad_docs))
        ranked_list2 = sorted(ranked_list2, key=lambda x: x[0]) 
        ranked_list2 = list(reversed(ranked_list))

    ranked_list = ranked_list + ranked_list2
    ranked_list = [i[1] for i in ranked_list]
        
    return ranked_list


#%%
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return "Main page of indexer"


@app.route('/indexer', methods=["POST"])
def add_to_index():
    """
    :param Document or list of Document documents:
    :return str: Status of adding.
    """
    global forward_index
    global inverted_index
    global documents_id
    
    documents = list(jsonpickle.decode(request.json))
    for document in documents:
        if str(document.id) not in documents_id:
            documents_id.append(str(document.id))
            forward_index = add_forward_index(document, forward_index)
            inverted_index = add_inverted_index(document, inverted_index)
            return "document is successfully added."
        else:
            return "document already exist in index."


@app.route("/search", methods=["POST"])
def search():
    """
    :param str search_query: must be normalized with stemming
    :return list of Documents search_result:                                             
    """
    search_query = request.json
    search_result = search_boolean(search_query, forward_index,
                                   inverted_index, documents_id)
    
    #search_result = ranking(search_result, search_query, vectorizer_tfidf)
    search_result = nn_rank_with_requirments(search_result, search_query, vectorizer_tfidf)
    
    return jsonpickle.encode(search_result[: 300])


@app.route("/save_index", methods=["POST"])
def saving():
    """Save current state of index."""
    
    save_index(config.index_dir, forward_index, inverted_index, documents_id)
    return "successfully saved."


#%%
if __name__ == "__main__":
    forward_index, inverted_index, documents_id = load_index(config.index_dir)
    vectorizer_tfidf = load_vectorizer_tfidf(config.index_dir,
                                             file="vectorizer_tfidf.dat")
    app.run(host=config.INDEXER_HOST, port=config.INDEXER_PORT)


