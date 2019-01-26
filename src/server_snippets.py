#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import json
import re

from flask import Flask, request
import jsonpickle

from config_src import config


#%%
def get_snippets():
    
    document = json.loads(request.json, encoding="utf-8")
    #exclude cases when only one-words query!!!!!
    documents = document["documents"]
    terms = []
    for term in document["terms"]:
        terms.append(term["inverted_index"][0]["pos"][0])
    
    result_lists = []
    for doc in documents:
        '''
        if terms[0] < 20:
            snippet = doc["text"][0:min(240, len(doc["text"]) - 1)]
        
        elif terms[0] > 20:
            snippet = doc["text"][terms[0]:min(terms[0]+240, len(doc["text"]) - 1)]
        '''
        snippet = doc["text"][0:len(doc["text"] - 1)]
            
        doc["snippet"] = doc["text"]    
    search_res = dict()
    search_res["results"] = documents
    return json.dumps(search_res, ensure_ascii=False)


#%%
app = Flask(__name__)


@app.route('/snippets', methods=["POST"])
def get_snippets():
    """
    :param dict params: Like {"documents": [list of Documents],
                              "terms":  list of dicts [{"term": "word1",
                                        "inverted_index": [dict1, dict2, ...]}]}
                              
    :return list of Documents documents: With updated snippet attributes
    """
    params = jsonpickle.decode(request.json)
    documents = params["documents"]
    query = params["query"]
    #query = " ".join([i["term"] for i in search_terms])
    
    for doc in documents:
        doc.snippet = doc.text[:min(240, len(doc.text))] 
    
    return jsonpickle.encode(documents)


#%%
if __name__ == "__main__":
    app.run(host=config.SNIPPETS_HOST, port=config.SNIPPETS_PORT)


#%%



