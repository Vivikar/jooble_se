import json
import gzip
import os

import requests
import pandas as pd

indexer_url = "http://127.0.0.1:13500/"
text_processing_url = "http://127.0.0.1:13501/"
ranking_url = "http://127.0.0.1:13502/"
snippets_url = "http://127.0.0.1:13503/"
res_page_form_url = "http://127.0.0.1:13504/"
query = "Саженцы культур"

print("Accessing indexer")
r = requests.post(indexer_url)
print(r.status_code)
print(r.text)

print("\nNormalizing document")
#  normalize the document
r = requests.post(text_processing_url + "normalize_document",
                  json={"id": 7, "title": "Садоводство",
                        "text": "Саженцы декоративных и плодовых культур. "
                                "Могилев. Гарантия."})
print(r.status_code)
print(json.loads(r.text))
document = json.loads(r.text, encoding="utf-8")
print("\nAdding document to index")

#  add the document to index
r = requests.post(indexer_url + "indexer", json=document)
print(r.status_code)
print(r.text)

print("\nNormalizing search query")
#  normalize the search query
r = requests.post(text_processing_url + "normalize_query",
                  json=query)
print(r.status_code)
print(str(r.text))
search_query = r.text

print("\nSearching in index")
# search in index
r = requests.post(indexer_url + "search",
                  json=search_query)
print(r.status_code)
print(r.text)
#search_results = type(json.loads(r.text, encoding="utf-8"))
search_results = r.text

print("\nGetting snippet-list")
# getting search results with snippets
h_query = r.text

r = requests.post(snippets_url + "snippets",
                  json=search_results)
print(r.status_code)
print(r.text)
srch_results = r.text

print("\nSERP results")
# getting search results with snippets
h_query = r.text

r = requests.post(res_page_form_url + "result_page",
                  json=h_query)
print(r.status_code)
print(r.text)

#--------##--------##--------##--------##--------##--------##--------##--------#

