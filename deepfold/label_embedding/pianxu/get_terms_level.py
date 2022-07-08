import os
from collections import Counter
import pandas as pd
import pickle
import numpy as np
from collections import deque
import random
import torch
from collections import defaultdict as ddt
from utils import Ontology, get_pairs


def main(triple_emb_file):
    # fake root 'GO:0000000'
    term_fake_root = 'GO:0000000'

    # three domains of BP, MF and CC
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {
        'cc': CELLULAR_COMPONENT,
        'mf': MOLECULAR_FUNCTION,
        'bp': BIOLOGICAL_PROCESS}

    NAMESPACES = {
        'cc': 'cellular_component',
        'mf': 'molecular_function',
        'bp': 'biological_process'
    }

    data_base_path = u"./data"

    ### INPUT FILES ###
    # Gene Ontology file in OBO Format
    obo_file = os.path.join(data_base_path, u"go.obo")

    data_path_dict = {}

    ### INPUT FILES ###
    data_path_dict['obo'] = obo_file

    for k, v in data_path_dict.items():
        print("{:}: {:} [{:5s}]".format(k, v, str(os.path.exists(v))))

    go = Ontology(data_path_dict['obo'], with_rels=True)

    with open("./data/terms_all.pkl", "rb") as fd:
        terms = pickle.load(fd)
        terms = list(terms["terms"])

    terms_set = set(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}

    terms_level_dict = ddt(int)
    def get_term_level(terms_set,terms_level_dict):
        if len(terms_set) == 0: return 0
        level = []
        for i in terms_set:
            if i not in terms_level_dict:
                terms_level_dict[i] = get_term_level(go.get_parents(i), terms_level_dict) + 1
                level.append(terms_level_dict[i])
            else:
                level.append(terms_level_dict[i])
        return max(level)
    get_term_level(terms, terms_level_dict)

    level_terms_dict = ddt(set)
    for k,v in terms_level_dict.items():
        level_terms_dict[v].add(k)
    level_terms_dict

    level_terms = []
    for i in range(2,19):
        level_terms.append(list(level_terms_dict[i]))
    train_data = []
    test_data = []
    for item in level_terms:
        random.shuffle(item)
        train_data += item[:int((len(item)+1)*.80)]
        test_data += item[int((len(item)+1)*.80):]

    train_terms_list = []
    test_terms_list = []
    for i in train_data:
        train_terms_list.append(terms_level_dict[i])
    for i in test_data:
        test_terms_list.append(terms_level_dict[i])

    model = torch.load(triple_emb_file, map_location="cpu")
    embeddings = model["embedding"].renorm_(2,0,1).numpy().tolist()

    train_embeddings_list = []
    test_embeddings_list = []
    for item in train_data:
        train_embeddings_list.append(embeddings[terms_dict[item]])
    for item in test_data:
        test_embeddings_list.append(embeddings[terms_dict[item]])
    test_embeddings_list

    train_terms = pd.DataFrame()
    test_terms = pd.DataFrame()
    train_terms['terms'] = train_data
    train_terms['embeddings'] = train_embeddings_list
    train_terms['labels'] = train_terms_list
    test_terms['terms'] = test_data
    test_terms['embeddings'] = test_embeddings_list
    test_terms['labels'] = test_terms_list

    # save
    train_terms.to_pickle('./data/train_terms.pkl')
    test_terms.to_pickle('./data/test_terms.pkl')

    
if __name__ == '__main__':
    # data folder must have go.obo file and terms_all.pkl file
    # terms_all.pkl file contains the terms ID.
    triple_emb_file = './data/triple_label_60.pth'  # 训练后保存下的文件
    main(triple_emb_file)