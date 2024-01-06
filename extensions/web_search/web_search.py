import requests
from extension import ExtensionInterface


class web_search(ExtensionInterface):

    def inference(self, newprompt, genparams, max_context, *args):
        responses = newprompt.split('### Response:')
        if len(responses) < 2:
            return newprompt
        response = responses[-1]
        if len(response) == 0:
            return newprompt
        print(f'response {response}')
        is_search = response.find('[search:')

        if is_search == -1:
            return newprompt

        query = response.split('[search:')[1].split(']')[0].strip()
        query = query.replace(' ', '+')
        print(f'query: {query}')
        json_object = requests.get(f'https://api.duckduckgo.com/?q={query}&format=json').json()
        result = json_object['Abstract'] if len(json_object['Abstract']) > 0 else json_object['RelatedTopics'][0]['Text']
        result_length = len(result)
        max_new_prompt = max_context-result_length

        newprompt = (newprompt[-(max_new_prompt):]) if len(newprompt) > max_new_prompt else newprompt
        return f'Search result:[{result}] {newprompt}'
