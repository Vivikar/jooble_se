#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import json

import requests
from flask import Flask, request, render_template
from wtforms import Form, TextField, validators, TextAreaField
import jsonpickle

from config_src import config
from document import Document
import text_processor

indexer_url = config.indexer_url
text_processing_url = config.text_processing_url
ranking_url = config.ranking_url
snippets_url = config.snippets_url
res_page_form_url = config.res_page_form_url


#%%
class ReusableForm(Form):
    query = TextField('Search Query: ', validators=[validators.required()])
    skills = TextAreaField('Enster skils: ')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    If GET -> renders Main page from templates/index.html
    If POST -> renders ranged search results
    """
    form = ReusableForm(request.form)
    if request.method == 'POST':
        query = request.form['query']
        skills = request.form["skills"]
        
        try:
            # Normilize search query
            search_query = text_processor.normalize_text(query)
            
            if search_query == "" or search_query == " " or search_query == []:
    
                search_result = {"documents": [], "query": query}
                r = requests.post(res_page_form_url + config.RESULT_PAGE_PATH,
                              json=jsonpickle.encode(search_result))
                result_page = r.text
                return r.text
                
            # Search
            r = requests.post(indexer_url + config.SEARCH_PATH,
                              json=search_query)
            search_result = jsonpickle.decode(r.text)
    
            #get rid off search_result=="Documents aren't found."
            if search_result == "" or search_result == " " or search_result == [] or search_result=="Documents aren't found.":
    
                search_result = {"documents": [], "query": query}
                r = requests.post(res_page_form_url + config.RESULT_PAGE_PATH,
                              json=jsonpickle.encode(search_result))
                return r.text
    
            documents = search_result
    
            # Ranking sould return list of extended docs
            search_result = {"documents": documents, "query": search_query, "skills": skills}
            r = requests.post(ranking_url + config.RANK_PATH,
                              json=jsonpickle.encode(search_result))
            search_result = jsonpickle.decode(r.text)
            documents = search_result[ :20]
    
            # Get snippets
            search_result = {"documents": documents,
                             "query": search_query}
            r = requests.post(snippets_url + config.SNIPPETS_PATH,
                              json=jsonpickle.encode(search_result))
            search_result = jsonpickle.decode(r.text)
    
            # SERP
            search_result = {"documents": search_result,
                             "query": query}
            r = requests.post(res_page_form_url + config.RESULT_PAGE_PATH,
                              json=jsonpickle.encode(search_result))
            result_page = r.text
            #if len(result_page == 0):
             #   result_page = None
            return result_page
        except:
            search_result = {"documents": [], "query": search_query}
            r = requests.post(res_page_form_url + config.RESULT_PAGE_PATH,
                                 json=jsonpickle.encode(search_result))
            return r.text       
    else:
        return render_template('main_page.html', form=form)


#%%
if __name__ == "__main__":
    app.run(host=config.MANAGER_HOST, port=config.MANAGER_PORT)


#%%



