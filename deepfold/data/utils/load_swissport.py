import argparse
import gzip
import logging
import sys

import pandas as pd

from deepfold.data.utils.data_utils import is_cafa_target, is_exp_code
from deepfold.data.utils.ontology import Ontology

sys.path.append('../../../')

logging.basicConfig(level=logging.INFO)


def load_swissport(swissprot_file):
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    interpros = list()
    orgs = list()
    with gzip.open(swissprot_file, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        ipros = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    interpros.append(ipros)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()
                ipros = list()
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    ipros.append(ipro_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        interpros.append(ipros)
        orgs.append(org)
    return proteins, accessions, sequences, annotations, interpros, orgs


parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--go-file',
                    '-gf',
                    default='data/go.obo',
                    help='Gene Ontology file in OBO Format')
parser.add_argument(
    '--swissprot-file',
    '-sf',
    default='data/uniprot_sprot.dat.gz',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
parser.add_argument(
    '--out-file',
    '-o',
    default='data/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')


def main(go_file, swissprot_file, out_file):
    go = Ontology(go_file, with_rels=True)

    proteins, accessions, sequences, annotations, interpros, orgs = load_swissport(
        swissprot_file)
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'annotations': annotations,
        'interpros': interpros,
        'orgs': orgs
    })

    logging.info('Filtering proteins with experimental annotations')
    index = []
    annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
        # Ignore proteins without experimental annotations
        if len(annots) == 0:
            continue
        index.append(i)
        annotations.append(annots)
    df = df.iloc[index]
    df = df.reset_index()
    df['exp_annotations'] = annotations

    prop_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            annot_set |= go.get_anchestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations

    cafa_target = []
    for i, row in enumerate(df.itertuples()):
        if is_cafa_target(row.orgs):
            cafa_target.append(True)
        else:
            cafa_target.append(False)
    df['cafa_target'] = cafa_target

    df.to_pickle(out_file)
    logging.info('Successfully saved %d proteins' % (len(df), ))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.go_file, args.swissprot_file, args.out_file)
