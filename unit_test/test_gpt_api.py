from api.gpt_api import get_chat_gpt_response, get_llama_gpt_response, get_spark_gpt_response

pre_prompt = "The assistant is helpful, creative, clever, and very friendly."
prompt = "Hello, who are you?"


def test_get_chat_gpt_response():
    response = get_chat_gpt_response(pre_prompt, prompt)
    print(response)
    assert response is not None


def test_get_baidu_gpt_response():
    response = get_llama_gpt_response(pre_prompt, prompt)
    print(response)
    assert response is not None


def test_get_spark_gpt_response():
    response = get_spark_gpt_response(pre_prompt, prompt)
    print(response)
    assert response is not None


if __name__ == '__main__':
    test_get_chat_gpt_response()
    test_get_baidu_gpt_response()
    test_get_spark_gpt_response()
