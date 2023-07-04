from extension import ExtensionInterface
import chromadb
import os
from os import listdir
from os.path import isfile, join


class chroma_db(ExtensionInterface):
    chromaCollection = None

    def __init__(self):
        self.chromaCollection = chromadb.Client().get_or_create_collection("koboldcpp")
        path = os.getcwd() + '/dbdata/'
        files = os.listdir(path)
        for f2 in files:
            if f2.endswith('.txt'):
                
                doc = open(path + f2).read()
                docs = doc.split('\n\n')
                print(f'found data file {f2} {len(docs)}')
                id_list = []
                meta_data_list = []
                for i in range(0, len(docs)):
                    id_list.append(f'{f2} {i}')
                    meta_data_list.append({'source': f'{f2}'})
                self.chromaCollection.add(documents=docs, metadatas=meta_data_list, ids=id_list)
        

    def inference(self, newprompt, genparams, max_context, *args):
        query_string = newprompt
        stop_sequence = genparams.get('stop_sequence', [])
        response_length = genparams.get('max_length', 50)
        max_distance = genparams.get('max_distance', 1.5) # TODO
        result_context_factor = genparams.get('result_context_factor', 0.25) #TODO
        n_results = genparams.get('n_results', 3) #TODO
        
        if stop_sequence:
            query_string = query_string.rsplit(stop_sequence[0], 1)[-1] or query_string
        # TODO: should shorten query anyway if no stop_sequence? or summarize?
        results = self.chromaCollection.query(query_texts=query_string, n_results=n_results)
        if not results:
            return newprompt
        trimmed_results = ''
        num_results = len(results['documents'][0])
        for i in range(0, num_results):
            distance = results['distances'][0][i]
            if distance < max_distance:
                bit = results['documents'][0][i]     
                # replacing line breaks, since they tend to stop the generation when double
                trimmed_results += bit.replace("\n", "")
                print(f' {distance} {bit[:15]}')
            else:
                print(f'not using {distance} {bit[:15]}')
        if not trimmed_results:
            return newprompt
        # max tokens for chromadb results is a fraction of max context
        result_max_tokens = int(max_context * result_context_factor)
        if len(newprompt) > max_context - result_max_tokens:
            result_max_tokens -= response_length
        #print(f'result_max_tokens: {result_max_tokens}')
        trimmed_results = (trimmed_results[:result_max_tokens]) if len(trimmed_results) > result_max_tokens else trimmed_results
        max_new_prompt = max_context-result_max_tokens
        newprompt = (newprompt[-(max_new_prompt):]) if len(newprompt) > max_new_prompt else newprompt
        return f'Topic:[{trimmed_results}] {newprompt}'
