import time

import openai

from api.gpt_api import get_chat_gpt_response, get_llama_gpt_response, get_spark_gpt_response


def get_completion(mask_sentence: object, count: object, model: object) -> object:
    pre_prompt = (
            "Let's play a word filling game, and I will give you a sentence where a word is covered by the [MASK] mark. "
            "You need to replace this mark with a suitable word and tell me the " + str(count) +
            " fully replaced sentences, one fully completed sentence per line. Please replay full sentence instead of just the word and don't add any meaningless marking symbols.")

    max_retries = 3
    retry_delay = 10

    for retry in range(max_retries):
        try:
            if model.lower() == "gpt":
                result = get_chat_gpt_response(pre_prompt, mask_sentence)
            elif model.lower() == "llama":
                result = get_llama_gpt_response(pre_prompt, mask_sentence)
            else:
                result = get_spark_gpt_response(pre_prompt, mask_sentence)
            result = result.split("\n")
            if result[0].startswith('1. '):
                result = [item.split(". ", 1)[1] for item in result]
            return result
        except openai.APITimeoutError:
            if retry == max_retries - 1:
                raise
            else:
                time.sleep(retry_delay)

    return None
