from api.parser_api import get_constituency_tree, get_dependency_tree


def test_get_constituency_tree(sentence):
    result = get_constituency_tree(sentence)
    print(result)
    assert result != ''


def test_get_dependency_tree(sentence):
    result = get_dependency_tree(sentence)
    print(result)
    assert result != ''


if __name__ == '__main__':
    sentence = "I am your awesome translate helper!"
    test_get_constituency_tree(sentence)
    test_get_dependency_tree(sentence)
