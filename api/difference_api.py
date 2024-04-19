import ast
from collections import defaultdict

import nltk
import spacy
from nltk.tree import Tree


class DependencyTree:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = set()
        for edge in edges:
            self.nodes.add(edge[1])
            self.nodes.add(edge[2])
        self.adj_list = defaultdict(list)
        for edge in edges:
            self.adj_list[edge[1]].append((edge[0], edge[2]))

    def get_dependency_path(self, node, visited=None):
        if visited is None:
            visited = set()
        if node == 'ROOT':
            return []
        if node in visited:
            # Avoid infinite loop in case of cycles
            return None
        visited.add(node)
        for dep, child in self.adj_list[node]:
            path = self.get_dependency_path(child, visited)
            if path is not None:
                return [dep] + path
        return None


def calculate_node_attribute_difference(tree1, tree2):
    differences = 0
    for node1, node2 in zip(tree1.subtrees(), tree2.subtrees()):
        if node1.label() != node2.label():
            differences += 1
    return differences


def get_feature_vector(con_tree1_str, con_tree2_str, dep_tree1_str, dep_tree2_str):
    tree1 = Tree.fromstring(con_tree1_str)
    tree2 = Tree.fromstring(con_tree2_str)

    dep_tree1 = DependencyTree(ast.literal_eval(dep_tree1_str.strip()))
    dep_tree2 = DependencyTree(ast.literal_eval(dep_tree2_str.strip()))

    nlp = spacy.load("en_core_web_sm")
    tree1_text = ' '.join([leaf for leaf in tree1.leaves()])
    tree2_text = ' '.join([leaf for leaf in tree2.leaves()])
    tree1_doc = nlp(tree1_text)
    tree2_doc = nlp(tree2_text)

    # Semantic similarity using spaCy
    semantic_distance = tree1_doc.similarity(tree2_doc)

    # Calculate differences in node attributes
    node_difference = calculate_node_attribute_difference(tree1, tree2)

    # Node changes
    node_changes = node_difference

    # POS changes
    pos_changes = defaultdict(int)
    for node1, node2 in zip(tree1.subtrees(), tree2.subtrees()):
        if node1.label() != node2.label():
            pos_changes[node1.label()] += 1

    # Dependency changes
    dep_changes = defaultdict(int)
    for node1, node2 in zip(tree1.subtrees(), tree2.subtrees()):
        if node1.label() in nltk.pos_tag([node1.label()])[0][1]:
            if nltk.pos_tag([node2.label()])[0][1] != nltk.pos_tag([node1.label()])[0][1]:
                dep_changes[nltk.pos_tag([node1.label()])[0][1]] += 1

    # Subtree changes
    subtree_changes = 0
    for node1, node2 in zip(tree1.subtrees(), tree2.subtrees()):
        if len(node1) != len(node2):
            subtree_changes += 1

    # Subtree depth changes
    subtree_depth_changes = sum(
        abs(node1.height() - node2.height()) for node1, node2 in zip(tree1.subtrees(), tree2.subtrees()))

    # Subtree width changes
    subtree_width_changes = sum(
        abs(len(node1) - len(node2)) for node1, node2 in zip(tree1.subtrees(), tree2.subtrees()))

    # Dependency edge changes
    dep_edge_changes = sum(1 for edge1 in dep_tree1.edges if edge1 not in dep_tree2.edges) + sum(
        1 for edge2 in dep_tree2.edges if edge2 not in dep_tree1.edges)

    # Dependency path changes
    dep_path_changes = sum(
        abs(len(node1) - len(node2)) for node1, node2 in zip(tree1.treepositions(), tree2.treepositions()) if
        tree1[node1] != tree2[node2])

    # Normalize features
    def normalize_feature(feature_value, min_value, max_value):
        return (feature_value - min_value) / (max_value - min_value)

    feature_vector = [
        normalize_feature(semantic_distance, -1, 1),
        normalize_feature(node_difference, 0, 100),
        normalize_feature(node_changes, 0, 100),
        normalize_feature(len(pos_changes), 0, 5),
        normalize_feature(len(dep_changes), 0, 5),
        normalize_feature(subtree_changes, 0, 100),
        normalize_feature(subtree_depth_changes, 0, 100),
        normalize_feature(subtree_width_changes, 0, 100),
        normalize_feature(dep_edge_changes, 0, 500),
        normalize_feature(dep_path_changes, 0, 100),
    ]

    return feature_vector
