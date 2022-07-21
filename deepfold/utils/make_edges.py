import math
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

from deepfold.data.utils.ontology import Ontology

sys.path.append('../')


# GOA_cnt
def statistic_terms(train_data_path):
    """get frequency dict from train file."""
    train_data = pd.read_pickle(train_data_path)
    cnt = Counter()
    for i, row in train_data.iterrows():
        for term in row['annotations']:
            cnt[term] += 1
    print('Number of annotated terms:', len(cnt))
    sorted_by_freq_tuples = sorted(cnt.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    freq_dict = {go: count for go, count in sorted_by_freq_tuples}
    return freq_dict


# make edges
def make_edges(go_file, namespace='bpo', with_rels=False,annotated_terms=None):
    go_ont = Ontology(go_file, with_rels=with_rels)
    if namespace == 'bpo':
        all_terms = go_ont.get_namespace_terms('biological_process')
    elif namespace == 'mfo':
        all_terms = go_ont.get_namespace_terms('molecular_function')
    elif namespace == 'cco':
        all_terms = go_ont.get_namespace_terms('cellular_component')
    
    if annotated_terms is not None:
        all_terms = all_terms.intersection(set(annotated_terms))
    edges = []
    for subj in all_terms:
        objs = go_ont.get_parents(subj)
        if len(objs) > 0:
            for obj in objs:
                if obj in all_terms:
                    edges.append((subj, obj))
    return edges


# make IC file
def read_go_children(input_go_obo_file):
    children = defaultdict(list)
    alt_id = defaultdict(list)
    term = False
    go_id = ''
    alt_ids = set()
    with open(input_go_obo_file) as read_in:
        for line in read_in:
            splitted_line = line.strip().split(':')
            if '[Term]' in line:
                term = True
                go_id = ''
                alt_ids = set()
            elif term and 'id: GO:' in line and 'alt_id' not in line:
                go_id = 'GO:{}'.format(splitted_line[2].strip())
            elif term and 'alt_id: GO' in line:
                alt_id_id = 'GO:{}'.format(splitted_line[2].strip())
                alt_ids.add(alt_id_id)
                alt_id[go_id].append(alt_id_id)
            elif term and 'is_a:' in line:
                splitted_term = splitted_line[2].split('!')
                go_term = 'GO:{}'.format(splitted_term[0].strip())
                children[go_term].append(go_id)
                for a in alt_ids:
                    children[go_term].append(a)
            elif '[Typedef]' in line:
                term = False
    return children, alt_id


def find_all_descendants(input_go_term, children):
    children_set = set()
    queue = []
    queue.append(input_go_term)
    while queue:
        node = queue.pop(0)
        if node in children and node not in children_set:
            node_children = children[node]
            queue.extend(node_children)
        children_set.add(node)
    return children_set


def store_counts_for_GO_terms(freq_dict, alt_id):
    go_cnt = defaultdict()
    for term in freq_dict:
        cnt = int(freq_dict[term])
        if term in alt_id.keys():
            for x in alt_id[term]:
                term = x
                if term not in go_cnt:
                    go_cnt[term] = cnt
                else:
                    go_cnt[term] = go_cnt[term] + cnt
        else:
            if term not in go_cnt:
                go_cnt[term] = cnt
            else:
                go_cnt[term] = go_cnt[term] + cnt
    return go_cnt


def calculate_freq(term, children_set, go_cnt):
    freq = 0
    if term in go_cnt.keys():
        freq = freq + go_cnt[term]
    for children in children_set:
        if children in go_cnt.keys():
            freq = freq + go_cnt[children]
    return freq


def calculate_information_contents_of_GO_terms(input_go_cnt_file, children,
                                               alt_id):
    ic_dict = defaultdict()
    go_cnt = store_counts_for_GO_terms(input_go_cnt_file, alt_id)
    for x in range(0, 3):
        if x == 0:
            root = 'GO:0005575'  # cellular component
        elif x == 1:
            root = 'GO:0008150'  # biological process
        elif x == 2:
            root = 'GO:0003674'  # molecular function
        root_descendants = find_all_descendants(root, children)
        root_freq = calculate_freq(root, root_descendants, go_cnt)
        for term in root_descendants:
            term_descendants = find_all_descendants(term, children)
            term_freq = calculate_freq(term, term_descendants, go_cnt)
            term_prob = (term_freq + 1) / (root_freq + 1)
            term_ic = -math.log(term_prob)
            assert (term not in ic_dict)
            ic_dict[term] = term_ic
    return ic_dict


# make final edge file
def get_all_go_cnt(edges, go_cnt, all_children, go_ic):
    all_go_cnt = []
    for children, parent in edges:
        cnt_every = 0.0
        cnt_chidren = 0.0
        cnt_freq_children = 0.0
        cnt_freq_parent = 0.0
        cnt_freq = 0.0

        if children in go_cnt.keys():
            cnt_freq_children = go_cnt[children]
        if parent in go_cnt.keys():
            cnt_freq_parent = go_cnt[parent]
        if cnt_freq_parent == 0.0 or cnt_freq_children == 0.0:
            cnt_freq = 1.0
        else:
            cnt_freq = cnt_freq_children / cnt_freq_parent

        if parent in all_children:
            for x in all_children[parent]:
                if x in go_ic.keys():
                    cnt_every += go_ic[x]

        if parent in go_ic.keys():
            cnt_every += go_ic[parent]
        if children in go_ic.keys():
            cnt_chidren += go_ic[children]

        cnt_every += cnt_chidren
        final_cnt = (cnt_chidren / cnt_every) + cnt_freq
        all_go_cnt.append((children, parent, final_cnt))
    return all_go_cnt


def get_go_ic(namespace='bpo', data_path=None):
    go_file = os.path.join(data_path, 'go_cafa3.obo')
    train_data_file = os.path.join(data_path, namespace,
                                   namespace + '_train_data.pkl')
    freq_dict = statistic_terms(train_data_file)
    annotated_terms = freq_dict.keys()
    edges = make_edges(go_file, namespace,False,annotated_terms)
    all_children, alt_id = read_go_children(go_file)
    go_ic = calculate_information_contents_of_GO_terms(freq_dict, all_children,
                                                       alt_id)
    all_go_cnt = get_all_go_cnt(edges, freq_dict, all_children, go_ic)
    return all_go_cnt


if __name__ == '__main__':
    data_path = '../../data/cafa3'
    all_go_bpo_cnt = get_go_ic('bpo', data_path)
    print(f'edges in bpo: {len(all_go_bpo_cnt)}')
    # print(f'nodes in bpo:{}')
    all_go_mfo_cnt = get_go_ic('mfo', data_path)
    print(f'edges in mfo: {len(all_go_mfo_cnt)}')
    all_go_cco_cnt = get_go_ic('cco', data_path)
    print(f'edges in cco: {len(all_go_cco_cnt)}')
