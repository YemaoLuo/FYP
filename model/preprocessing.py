import hashlib
import os
import warnings

import joblib
import pandas as pd
from tqdm import tqdm

from api.difference_api import get_feature_vector
from api.parser_api import get_constituency_tree, get_dependency_tree

warnings.filterwarnings("ignore")


def prepare_data():
    CACHE_DIR = "./cache/"
    CACHE_DIR_2 = "../tree/"
    df = pd.read_excel('./data.xlsx')
    excel_data = df.iloc[:, 0:4].values.tolist()
    vector_data = []
    for row in tqdm(excel_data):
        sentence1 = row[0]
        sentence2 = row[1]
        label = row[2]

        # Hash the sentences to use as cache filename for feature vector
        cache_filename = hashlib.sha256((sentence1 + sentence2).encode()).hexdigest() + ".joblib"
        cache_filepath = os.path.join(CACHE_DIR, cache_filename)

        if os.path.exists(cache_filepath):
            # If the cached feature vector exists, load it
            vector = joblib.load(cache_filepath)
        else:
            # Check if trees are cached
            tree_cache_filename_1 = hashlib.sha256(sentence1.encode()).hexdigest() + ".joblib"
            tree_cache_filepath_1 = os.path.join(CACHE_DIR_2, tree_cache_filename_1)
            if os.path.exists(tree_cache_filepath_1):
                con_tree_1, dep_tree_1 = joblib.load(tree_cache_filepath_1)
            else:
                con_tree_1 = get_constituency_tree(sentence1)
                dep_tree_1 = get_dependency_tree(sentence1)
                joblib.dump((con_tree_1, dep_tree_1), tree_cache_filepath_1)

            tree_cache_filename_2 = hashlib.sha256(sentence2.encode()).hexdigest() + ".joblib"
            tree_cache_filepath_2 = os.path.join(CACHE_DIR_2, tree_cache_filename_2)
            if os.path.exists(tree_cache_filepath_2):
                con_tree_2, dep_tree_2 = joblib.load(tree_cache_filepath_2)
            else:
                con_tree_2 = get_constituency_tree(sentence2)
                dep_tree_2 = get_dependency_tree(sentence2)
                joblib.dump((con_tree_2, dep_tree_2), tree_cache_filepath_2)

            # Compute the feature vector and save it to cache
            vector = get_feature_vector(con_tree_1, con_tree_2, dep_tree_1, dep_tree_2)
            joblib.dump(vector, cache_filepath)

        vector_data.append([vector, label])

    return vector_data


if __name__ == '__main__':
    data = prepare_data()
    # Write data to Excel
    df = pd.DataFrame(data, columns=['feature_vector', 'label'])
    df.to_excel('data_vector.xlsx', index=False)
