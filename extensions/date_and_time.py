from extension import ExtensionInterface
import time

class date_and_time(ExtensionInterface):
    
    def inference(self, newprompt, genparams, max_context, *args):
        responses = newprompt.split('###Response')
        if not responses:
            return newprompt
        response = responses[-1]
        date_and_time = response.find('[date_time]')
        if not date_and_time:
            return newprompt

        time_string = time.ctime()
        
        result_length = len(time_string)
        max_new_prompt = max_context-result_length
        newprompt = (newprompt[-(result_length):]) if len(newprompt) > max_new_prompt else newprompt
        return f'Date and time:[{time_string}] {newprompt}'
        
        
        
