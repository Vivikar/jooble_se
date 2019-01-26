#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import json
import pickle
import os
import re
import json

from flask import Flask, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jsonpickle
import pandas as pd

from config_src import config
from document import Document


#%%
def load_vectorizer_tfidf(path, file="vectorizer_tfidf.dat"):
    """
    Load tfidf vectorizer from file.
    
    If files don't exist - error.

    :return TfidfVectorizet vectorizer:
    """ 
    file_path = os.path.join(path, file)
    
    file_present = os.path.exists(file_path)
  
    if file_present:    
        with open(file_path, "rb") as inf:
            vectorizer = pickle.load(inf)
        return vectorizer
    else:
        print("File vectorizer not found.")
    return None


def nn_rank(documents, skills, vectorizer):
    skills_vect = vectorizer.transform([skills]).todense()
    doc_vects = [doc.requirement_normalized for doc in documents]
    doc_vects = vectorizer.transform(doc_vects).todense()
    
    ranked_list = cosine_similarity(doc_vects, skills_vect)
    ranked_list = list(np.squeeze(ranked_list, axis=1))
    assert(len(ranked_list) == len(documents))
    
    ranked_list = list(zip(ranked_list, documents))
    ranked_list = sorted(ranked_list, key=lambda x: x[0]) 
    ranked_list = list(reversed(ranked_list))
    #print(ranked_list)
    return ranked_list


def nn_rank_prof_area(documents, query, vectorizer):
    query_vect = vectorizer.transform([query]).todense()
    doc_vects = [doc.prof_area_normalized for doc in documents]
    doc_vects = vectorizer.transform(doc_vects).todense()
    
    ranked_list = cosine_similarity(doc_vects, query_vect)
    ranked_list = list(np.squeeze(ranked_list, axis=1))
    assert(len(ranked_list) == len(documents))
    
    ranked_list = list(zip(ranked_list, documents))
    ranked_list = sorted(ranked_list, key=lambda x: x[0]) 
    ranked_list = list(reversed(ranked_list))
    #print(ranked_list)
    return ranked_list


#%%
app = Flask(__name__)


@app.route('/ranking', methods=["POST"])
def ranking():
    params = jsonpickle.decode(request.json)
    documents = params["documents"]
    skills = params["skills"]
    query = params["query"]
      
    if skills != " ":
        ranked_list = nn_rank(documents, skills, vectorizer_tfidf)
    else:
        ranked_list = nn_rank_prof_area(documents, query, vectorizer_tfidf)
  
    ranked_list = [i[1] for i in ranked_list]

    return jsonpickle.encode(ranked_list)


#%%
if __name__ == "__main__":
    vectorizer_tfidf = load_vectorizer_tfidf(config.index_dir)
    
    app.run(host=config.RANKING_HOST, port=config.RANKING_PORT)


