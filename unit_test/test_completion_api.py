from api.completion_api import get_completion


def test_get_completion(model):
    sentence = "The assistant is [MASK], creative, clever, and very friendly."
    count = 5
    return get_completion(sentence, count, model)


if __name__ == '__main__':
    completion = test_get_completion("spark")
    print(completion)
