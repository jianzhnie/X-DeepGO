from collections import deque
import math
import os
from collections import Counter
import pandas as pd
import pickle
import numpy as np
from copy import deepcopy as dc


# Gene Ontology based on .obo File
# 修改版
class Ontology(object):

    def __init__(self, filename='./data/go.obo', with_rels=False):
        super().__init__()
        self.ont, self.format_version, self.data_version = self.load(filename, with_rels)
        self.ic = None
    
    # ------------------------------------
    def load(self, filename, with_rels):
        ont = dict()
        format_version = []
        data_version = []
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    # format version line
                if line.startswith('format-version:'):
                    l = line.split(": ")
                    format_version = l[1]
                # data version line
                if line.startswith('data-version:'):
                    l = line.split(": ")
                    data_version = l[1]
                # item lines
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    # four types of relations to others: is a, part of, has part, or regulates
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['relationship'] = list()
                    # alternative GO term id
                    obj['alt_ids'] = list()
                    # is_obsolete
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships revised
                        if it[0] == 'part_of':
                            obj['part_of'].append(it[1])
                        obj['relationship'].append([it[1], it[0]])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        # dealing with alt_ids, why?
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        # is_a -> children
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a'] + val['part_of']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont, format_version, data_version
       
    # ------------------------------------
    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]
    
    # revised 'part_of'
    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in (self.ont[t_id]['is_a'] + self.ont[t_id]['part_of']):
                    if parent_id in self.ont:
                        q.append(parent_id)
        # terms_set.remove(term_id)
        return term_set

    # revised
    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in (self.ont[term_id]['is_a'] + self.ont[term_id]['part_of']):
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set
    
    # get the root terms(only is_a)
    def get_root_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        # terms_set.remove(term_id)
        return term_set

    def get_roots(self, term_id):
        if term_id not in self.ont:
            return set()
        root_set = set()
        for term in self.get_root_ancestors(term_id):
            if term not in self.ont:
                continue
            if len(self.get_parents(term)) == 0:
                root_set.add(term)
        
        return root_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    # all children
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set
    
    # only one layer children
    def get_child_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        if term_id not in term_set:
            for ch_id in self.ont[term_id]['children']:
                term_set.add(ch_id)
        return term_set


def get_pairs(terms, terms_dict, ancestor_dict, parents_dict, child_dict, root_dict):
    
    n_cc = ["GO:0005575",'GO:0032991','GO:0044423','GO:0110165','GO:0019013']
    n_bp = ["GO:0008150",'GO:0000003','GO:0002376','GO:0006791','GO:0006794']
    n_mf = ['GO:0003674','GO:0003774','GO:0003824','GO:0005198','GO:0005215']
    pair_rank = list()
    temp_list = [terms_dict[terms], -1, -1]
    parent_list, child_list, root_list = list(), list(), list()
    for item in ancestor_dict[terms]:
        temp_list[1] = terms_dict[item]
        if parents_dict[item] is not None:
            for j in parents_dict[item]:
                temp_list2 = dc(temp_list)
                temp_list2[2] = terms_dict[j]
                parent_list.append(temp_list2)
        if child_dict[item] is not None:
            for j in child_dict[item]:
                if j not in ancestor_dict[terms] and j != terms:
                    temp_list2 = dc(temp_list)
                    temp_list2[2] = terms_dict[j]
                    child_list.append(temp_list2)
    pair_rank.append(parent_list)
    pair_rank.append(child_list)
    if root_dict[terms] is not None:
        temp_root_set = {"GO:0005575", "GO:0008150", "GO:0003674"}
        temp_root_set.remove(root_dict[terms])
        temp_list[1] = terms_dict[root_dict[terms]]
        if root_dict[terms] == 'GO:0005575':
            for k in n_bp + n_mf:
                temp_list2 = dc(temp_list)
                temp_list2[2] = terms_dict[k]
                root_list.append(temp_list2)
        elif root_dict[terms] == 'GO:0003674':
            for k in n_bp + n_cc:
                temp_list2 = dc(temp_list)
                temp_list2[2] = terms_dict[k]
                root_list.append(temp_list2)
        elif root_dict[terms] == 'GO:0008150':
            for k in n_mf + n_cc:
                temp_list2 = dc(temp_list)
                temp_list2[2] = terms_dict[k]
                root_list.append(temp_list2)
    pair_rank.append(root_list)
    return pair_rank