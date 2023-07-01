import chromadb
import os
from os import listdir
from os.path import isfile, join

chromaCollection = None

def init_chromadb():
    global chromaCollection
    chromaCollection = chromadb.Client().get_or_create_collection("koboldcpp")
    files = os.listdir(os.getcwd() + '/dbdata')
    print(f'found files in directory {files}')
    for f2 in files:
        #print('found file ' % f2.name)
        if f2.endswith('txt'):
            print(f'found data file {f2} ')                
            chromaCollection.add(documents=open(os.getcwd() + '/dbdata/' + f2).read(), metadatas={'source': f'{f2}'}, ids=f2)
    

def query_chromadb(newprompt, stop_sequence, maxctx):
    query_string = newprompt
    if stop_sequence:
        query_string = query_string.rsplit(stop_sequence[0], 1)[-1] or query_string
    print(f'query_string: {query_string}    ')
    results = chromaCollection.query(query_texts=query_string, n_results=1)
    if results:
        trimmedResults = results['documents'][0][0].replace("\n", "")
        maxLength = (trimmedResults[:512] + '..') if len(trimmedResults) > 512 else trimmedResults
        newprompt = (newprompt[-(maxctx-512):] + '..') if len(newprompt) > maxctx-512 else newprompt
        newprompt = f'Topic memory:[{trimmedResults}] {newprompt}'
        print(f'chromadb result found {newprompt}')
    else:
        print('no chromadb result for topic')
    return newprompt
