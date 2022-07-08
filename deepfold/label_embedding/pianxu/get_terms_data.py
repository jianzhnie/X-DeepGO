import os
from collections import Counter
import pandas as pd
import pickle
import numpy as np
from collections import deque
import math
from collections import defaultdict as ddt
from utils import Ontology, get_pairs


def main():
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

    child_dict = ddt(set)
    for i in range(len(terms)):
        child_dict[terms[i]] = terms_set.intersection(go.get_child_set(terms[i]))

    parents_dict = ddt(set)
    for i in range(len(terms)):
        parents_dict[terms[i]] = terms_set.intersection(go.get_parents(terms[i]))

    ancestor_dict = ddt(set)
    for i in range(len(terms)):
        temp_set = go.get_ancestors(terms[i])
        temp_set.remove(terms[i])
        ancestor_dict[terms[i]] = terms_set.intersection(temp_set)

    root_set = {"GO:0005575", "GO:0008150", "GO:0003674"}
    root_dict = ddt(set)
    for i in range(len(terms)):
        root_dict[terms[i]] = go.get_roots(terms[i])

    # adjusting root_dict
    for k,v in root_dict.items():
        root_dict[k] = list(v)[0]

    # 选取前10个作为正样本
    contrast_dict = ddt(set)
    contrast_dict['p_cc'] = {"GO:0005575",'GO:0032991','GO:0044423','GO:0110165','GO:0019013','GO:0019028','GO:0019033','GO:0036338','GO:0039624','GO:0039625'}
    contrast_dict['p_mf'] = {'GO:0003674','GO:0003774','GO:0003824','GO:0005198','GO:0005215','GO:0005488','GO:0016209','GO:0031386','GO:0038024','GO:0044183'}
    contrast_dict['p_bp'] = {"GO:0008150",'GO:0000003','GO:0002376','GO:0006791','GO:0006794','GO:0007610','GO:0008152','GO:0009758','GO:0009987','GO:0015976'}
    bp, cc, mf = set(), set(), set()
    for k,v in root_dict.items():
        if k == "GO:0005575" or "GO:0005575" in v:
            cc.add(k)
        elif k == 'GO:0003674' or 'GO:0003674' in v:
            mf.add(k)
        elif k == "GO:0008150" or "GO:0008150" in v:
            bp.add(k)
    contrast_dict['n_cc'] = cc
    contrast_dict['n_bp'] = bp
    contrast_dict['n_mf'] = mf

    # 构造对比学习部分的数据集
    contrast_dict = {**contrast_dict, **root_dict}

    pair_list = list()
    for i in range(len(terms)):
        pair_list.append(get_pairs(terms[i], terms_dict, ancestor_dict, parents_dict, child_dict, root_dict))

    # save data
    with open("./data/pairs_all.pkl", "wb")as fd:
        pickle.dump(pair_list, fd)

    with open("./data/contrast_pairs_all.pkl", "wb")as fd:
        pickle.dump(contrast_dict, fd)


if __name__ == '__main__':
    # data folder must have go.obo file and terms_all.pkl file
    # terms_all.pkl file contains the terms ID.
    main()