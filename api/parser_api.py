import os

from stanfordcorenlp import StanfordCoreNLP

dist = os.getcwd()
while dist.split('\\')[-1] != 'StructureTester':
    dist = "\\".join(dist.split('\\')[:-1])
dist += '\\stanford-corenlp-4.5.5'


def get_constituency_tree(sentence: str):
    nlp = StanfordCoreNLP(dist)
    res = nlp.parse(sentence)
    nlp.close()
    return str(res)


def get_dependency_tree(sentence: str):
    nlp = StanfordCoreNLP(dist)
    res = nlp.dependency_parse(sentence)
    nlp.close()
    return str(res)
