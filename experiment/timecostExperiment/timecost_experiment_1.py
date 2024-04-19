from api.characteristic_api import *
from api.completion_api import *
from api.difference_api import *
from api.parser_api import *
from api.translate_api import *

if __name__ == '__main__':
    sentence = "This is a test for time cost."
    st = time.thread_time()
    print(baidu_translate(sentence, 'en', 'zh'))
    print("Baidu translate time cost: ", time.thread_time() - st)
    st = time.thread_time()
    print(google_translate(sentence, 'en', 'zh-CN'))
    print("Google translate time cost: ", time.thread_time() - st)

    st = time.thread_time()
    c = get_constituency_tree(sentence)
    print(c)
    print("Constituency tree time cost: ", time.thread_time() - st)
    st = time.thread_time()
    d = get_dependency_tree(sentence)
    print(d)
    print("Dependency tree time cost: ", time.thread_time() - st)

    st = time.thread_time()
    print(get_feature_vector(c, c, d, d))
    print("Feature vector time cost: ", time.thread_time() - st)

    st = time.thread_time()
    print(get_chat_gpt_response("Try to create 10 sentences in Chinese line by line.", ""))
    print("Chat GPT time cost: ", time.thread_time() - st)

    st = time.thread_time()
    m = get_mask_sentence(sentence)
    print(m)
    print("Mask sentence time cost: ", time.thread_time() - st)

    st = time.thread_time()
    print(get_completion(m, 10, 'gpt'))
    print('Completion time cost: ', time.thread_time() - st)
