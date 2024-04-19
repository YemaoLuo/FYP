from api.difference_api import get_feature_vector
from api.parser_api import *


def test_get_difference(t1, t2, t3, t4):
    feature_vector = get_feature_vector(t1, t2, t3, t4)
    print(feature_vector)


if __name__ == '__main__':
    sentence1 = "This impressive literary work is renowned for its profound content and beautiful language, hailed as a classic of contemporary literature."
    sentence2 = "This impressive literary work is renowned for its profound content and beautiful language, hailed as a classic of contemporary literature."

    tree1 = get_constituency_tree(sentence1)
    tree2 = get_constituency_tree(sentence2)
    tree3 = get_dependency_tree(sentence1)
    tree4 = get_dependency_tree(sentence2)
    test_get_difference(tree1, tree2, tree3, tree4)

# [0.9212154158065465, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 5, 9, 3, 14]
# 1. **语义/情感特征 (0.9212154158065465):**
#    - 这个特征使用了 spaCy 的语义相似度计算方法,通过比较两个句子的语义表示来得到它们的相似度分数。
#    - 这个特征可以反映两个句子在语义上的相似程度,对于文本相似度评估、文本聚类等任务非常有帮助。语义相似度越高,说明两个句子的含义越接近。
#
# 2. **节点属性差异特征 (3):**
#    - 这个特征统计了两个句法树中节点标签(如词性、依存关系等)不同的数量。
#    - 这个特征可以反映两个句子在词汇和语法结构上的差异程度,对于分析句子的语法复杂度和句式差异很有帮助。节点属性差异越大,说明两个句子在语法结构上的差异也越大。
#
# 3. **词性(POS)变化特征 (2, 0, 0, 0, 0):**
#    - 这个特征统计了两个句法树中不同词性标签的数量变化,其中包括 `nsubj`、`obj`、`amod`、`advmod` 和 `compound` 等常见的词性标签。
#    - 这个特征可以反映两个句子在词性分布上的差异,对于分析句子的语法复杂度、信息结构等方面很有帮助。词性变化越大,说明两个句子在语法结构上的差异也越大。
#
# 4. **子树结构变化特征 (0, 0, 0):**
#    - 这个特征包括子树数量变化、子树深度变化和子树宽度变化,用于量化两个句法树在结构上的差异。
#    - 这些特征可以反映两个句子在句法复杂度和信息密集程度上的差异,对于文本难易程度分析、自动文本摘要等任务很有帮助。子树结构变化越大,说明两个句子的句法复杂度差异也越大。
#
# 5. **依存关系边变化特征 (10):**
#    - 这个特征统计了两个依存分析树中依存关系边的变化数量,即有多少个依存关系边在两个树中不同。
#    - 这个特征可以反映两个句子在依存关系结构上的差异,对于分析句子的语义角色和信息结构很有帮助。依存关系边变化越大,说明两个句子的语义结构差异也越大。
#
# 6. **依存关系路径变化特征 (11):**
#    - 这个特征统计了两个依存分析树中依存关系路径的变化数量,即有多少个节点到根节点的依存关系路径在两个树中不同。
#    - 这个特征可以反映两个句子在语义角色和信息结构上的差异,对于分析句子的复杂性和重要性很有帮助。依存关系路径变化越大,说明两个句子的语义结构差异也越大。
#
# 7-9. **依存树路径差异特征 (5, 9, 3):**
#    - 这些特征量化了两个依存分析树中节点到根节点路径长度的差异。包括路径变化数量、平均路径长度差异和最大路径长度差异。
#    - 这些特征可以反映两个句子在信息密集程度和语义结构复杂性上的差异,对于分析句子的重要性和难易程度很有帮助。路径差异越大,说明两个句子的语义结构差异也越大。
#
# 10. **依存关系边变化特征 (14):**
#     - 这个特征统计了两个依存分析树中依存关系边的总变化数量。
#     - 这个特征可以综合反映两个句子在依存关系结构上的差异,对于分析句子的语义角色和信息结构很有帮助。依存关系边变化越大,说明两个句子的语义结构差异也越大。
