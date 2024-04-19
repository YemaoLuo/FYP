import hashlib
import json
import os
import random

import requests


# Need to set environment variables first
def baidu_translate(query: str, from_lang: str, to_lang: str):
    appid = os.environ.get('BAIDU_TRANS_APPID')
    secret_key = os.environ.get('BAIDU_TRANS_SECRETKEY')
    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    salt = random.randint
    sign = appid + query + str(salt) + secret_key
    sign = hashlib.md5(sign.encode()).hexdigest()

    params = {
        'q': query,
        'from': from_lang,
        'to': to_lang,
        'appid': appid,
        'salt': salt,
        'sign': sign
    }

    response = requests.get(url, params=params)
    result = json.loads(response.text)
    if 'trans_result' in result:
        translations = result['trans_result']
        translated_text = translations[0]['dst']
        return translated_text
    else:
        return result


def google_translate(query: str, from_lang: str, to_lang: str):
    url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&q={}'.format(from_lang,
                                                                                                        to_lang, query)
    response = requests.get(url)
    result = json.loads(response.text)
    translated_text = result[0][0][0]
    return translated_text
