import random

import spacy


# Install require model first:
# python -m spacy download zh_core_web_sm
# python -m spacy download en_core_web_sm
def get_mask_sentence(sentence):
    pos_list = ['NOUN', 'VERB', 'ADJ']
    if is_chinese(sentence):
        nlp = spacy.load('zh_core_web_sm')
    else:
        nlp = spacy.load('en_core_web_sm')

    doc = nlp(sentence)
    words = [token.text_with_ws for token in doc]
    pos_tags = [token.pos_ for token in doc]
    pos_indices = [i for i, tag in enumerate(pos_tags) if tag in pos_list]
    random.shuffle(pos_indices)

    for index in pos_indices:
        words[index] = '[MASK]'
        new_sentence = ''.join(words)
        if not is_chinese(sentence) and new_sentence[-2] != ']':
            new_sentence = new_sentence.replace(']', '] ')
        return new_sentence

    return None


def is_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False
