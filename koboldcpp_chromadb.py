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
        if f2.endswith('txt'):
            
            doc = open(os.getcwd() + '/dbdata/' + f2).read()
            docs = doc.split('\n\n')
            print(f'found data file {f2} {len(docs)}')
            # TODO: fix proper ids
            id_list = []
            meta_data_list = []
            for i in range(0, len(docs)):
                id_list.append(f'{f2} {i}')
                meta_data_list.append({'source': f'{f2}'})
            chromaCollection.add(documents=docs, metadatas=meta_data_list, ids=id_list)
    

def query_chromadb(newprompt, stop_sequence, maxctx, response_length, max_distance = 1.5, result_context_factor = 0.25, n_results = 3):
    query_string = newprompt
    if stop_sequence:
        query_string = query_string.rsplit(stop_sequence[0], 1)[-1] or query_string
    # should shorten query anyway if no stop_sequence?
    #print(f'query_string: {query_string}')
    # only pick top result for now
    results = chromaCollection.query(query_texts=query_string, n_results=n_results)
    if not results:
        print('no chromadb result for topic')
        return newprompt
    # replacing line breaks, since they tend to stop the generation when doubled
    trimmed_results = ''
    num_results = len(results['documents'][0])
    print(f'results {num_results}')
    for i in range(0, num_results):
        distance = results['distances'][0][i]
        if distance < max_distance:
            bit = results['documents'][0][i]
            trimmed_results += bit.replace("\n", "")
            #print(f'result{i} {distance} {bit}')

    if not trimmed_results:
        return newprompt
    # max tokens for chromadb results is either max context length - current context, or a fraction of max context
    result_max_tokens = max(int(maxctx * result_context_factor), maxctx-len(newprompt))
    if len(newprompt) > maxctx - result_max_tokens:
        # only shorten the result if the prompt is long enough
        result_max_tokens -= response_length
    #print(f'result_max_tokens: {result_max_tokens}')
    trimmed_results = (trimmed_results[:result_max_tokens]) if len(trimmed_results) > result_max_tokens else trimmed_results
    max_new_prompt = maxctx-result_max_tokens
    newprompt = (newprompt[-(max_new_prompt):]) if len(newprompt) > max_new_prompt else newprompt
    #print(f'newprompt length: {len(newprompt)}')
    newprompt = f'Topic:[{trimmed_results}] {newprompt}'
    #print(f'chromadb result found {newprompt} {distance}')
    return newprompt
