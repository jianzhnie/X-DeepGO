import pickle

import anc2vec
import anc2vec.train as builder
import tempfile
import tensorflow as tf
from anc2vec.train.utils import Tokenizer
from anc2vec.train import onto
import numpy as np
import pandas as pd


# 3.3550
def anc_train(go_obo_file='./data/go.obo'):
    es = builder.fit(go_obo_file, embedding_sz=200, batch_sz=64, num_epochs=100)

def load_anc_emb(go_obo_file='./data/go.obo'):
    model_name = '/tmp/models/' + 'Embedder_embedding_sz=200' + '/best.tf'
    model = tf.keras.models.load_model(model_name, compile=False)
    embeddings = model.get_layer('embedding').weights[0].numpy()
    # transform embeddings into a dictionary
    embeds = {}
    go = onto.Ontology(go_obo_file, with_rels=True, include_alt_ids=False)
    tok = Tokenizer(go)
    for i, t in enumerate(tok.term2index):
        embeds[t] = embeddings[i, :]
    return embeds

def get_target_terms(embeds, in_terms_file="./data/terms.pkl", out_terms_file="./data/terms_emb_200_pd.pkl"):
    terms_emb = np.zeros((5874, 200), dtype=np.float32)
    # Terms of annotations
    terms_df = pd.read_pickle(in_terms_file)
    terms = terms_df['terms'].values.flatten()
    print(terms)
    nb_classes = len(terms)
    print("Number of classes: ", nb_classes, "\n")
    for i in range(len(terms)):
        if terms[i] == 'GO:0019012':
            terms_emb[i] = embeds['GO:0044423']
        elif terms[i] == 'GO:0007050':
            terms_emb[i] = embeds['GO:0051726']
        elif terms[i] == 'GO:0033613' or terms[i] == 'GO:0070491' or terms[i] == 'GO:0001107':
            terms_emb[i] = embeds['GO:0140297']
        elif terms[i] == 'GO:0001102' or terms[i] == 'GO:0001085' or terms[i] == 'GO:0001103':
            terms_emb[i] = embeds['GO:0061629']
        elif terms[i] == 'GO:0000778':
            terms_emb[i] = embeds['GO:0000776']
        else:
            terms_emb[i] = embeds[terms[i]]

    final_frame = pd.DataFrame()
    final_frame['Terms'] = terms
    final_frame['embedding'] = terms_emb.tolist()
    print(final_frame)
    pd.to_pickle(final_frame, out_terms_file)
    # with open("../data/terms_emb_200.pkl", 'wb') as fd:
    #     pickle.dump(terms_emb, fd)

    # # 把embedding加载到pkl里
    # with open("./data/terms_emb_200.pkl", 'rb') as fd:
    #     a = pickle.load(fd)
    #     print(a.shape)

if __name__ == '__main__':
    anc_train()
    embs = load_anc_emb()
    get_target_terms(embs)