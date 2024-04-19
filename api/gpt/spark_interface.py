import os

from api.gpt import SparkApi

appid = os.getenv("SPARK_APPID")
api_secret = os.getenv("SPARK_API_SECRET")
api_key = os.getenv("SPARK_API_KEY")

domain = "generalv2"
Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"

text = []


# length = 0

def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text


def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length


def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text


def get_spark_gpt_response_interface(pre_prompt: str, prompt: str):
    text.clear()
    question = checklen(getText("user", pre_prompt))
    SparkApi.answer = ""
    SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
    getText("assistant", SparkApi.answer)
    question = checklen(getText("user", prompt))
    SparkApi.answer = ""
    SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
    getText("assistant", SparkApi.answer)
    return str(text[-1]["content"])
