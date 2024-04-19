import json
import os

import requests
from openai import OpenAI

from api.gpt.spark_interface import get_spark_gpt_response_interface


# Need to set environment variables first
def get_chat_gpt_response(pre_prompt: str, prompt: str):
    client = OpenAI()
    client.api_key = os.environ.get("OPENAI_API_KEY")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content


# Need to set environment variables first
def get_llama_gpt_response(pre_prompt: str, prompt: str):
    api_key = os.environ.get("BAIDU_GPT_API_KEY")
    secret_key = os.environ.get("BAIDU_GPT_SECRET_KEY")
    url = ("https://aip.baidubce.com/oauth/2.0/token"
           "?grant_type=client_credentials&client_id=" + api_key + "&client_secret=" + secret_key)

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    url = ("https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/qianfan_chinese_llama_2_13b"
           "?access_token=") + response.json().get("access_token")

    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": pre_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def get_spark_gpt_response(pre_prompt: str, prompt: str):
    return get_spark_gpt_response_interface(pre_prompt, prompt)
