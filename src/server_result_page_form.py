#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import json

from flask import Flask, request, render_template
import jsonpickle
import requests

from config_src import config


#%%
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

class ReusableForm(Form):
    query = TextField('Search Query: ', validators=[validators.required()])
    skills = TextAreaField('Enster skils: ')


#%%
app = Flask(__name__)

@app.route('/result_page', methods=["POST"])
def get_result_page():
    """
    :param dict params: Like {"documents": [list of Documents],
                              "query": str query}
                              
    :return str serp: Search engine result page
    """
    params = jsonpickle.decode(request.json)
    form = ReusableForm(request.form)
    if params["documents"] == []:
        return render_template('result_page.html', form=form, documents=None, query=params["query"])
        
    return render_template('result_page.html', form=form, documents = params["documents"])


#%%
if __name__ == "__main__":
    app.run(host=config.RESULT_PAGE_HOST,
            port=config.RESULT_PAGE_PORT)


#%%



