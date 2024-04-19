from api.characteristic_api import get_mask_sentence


def test_get_mask_sentence():
    res = get_mask_sentence(sentence)
    print(res)
    assert res != '' or res is None


if __name__ == '__main__':
    sentence = "I love this beautiful view from the mountains。"
    test_get_mask_sentence()
    sentence = "选举结果使许多人感到惊讶，因为处于劣势的候选人获胜了。"
    test_get_mask_sentence()
