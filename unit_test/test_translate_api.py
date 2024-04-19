from api.translate_api import baidu_translate, google_translate


def test_baidu_translate():
    query = 'This is a sentence to baidu translate.'
    from_lang = 'en'
    to_lang = 'zh'
    translated_text = baidu_translate(query, from_lang, to_lang)
    print(translated_text)
    assert translated_text != ''


def test_google_translate():
    query = 'This is a sentence to google translate.'
    from_lang = 'en'
    to_lang = 'zh-CN'
    translated_text = google_translate(query, from_lang, to_lang)
    print(translated_text)
    assert translated_text != ''


if __name__ == '__main__':
    test_baidu_translate()
    test_google_translate()
