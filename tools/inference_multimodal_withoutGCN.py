import argparse
import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import yaml
from torch.utils.data import DataLoader

from deepfold.data.gcn_dataset import GCNDataset
from deepfold.models.esm_model import MLP
from deepfold.trainer.training import predict
from deepfold.utils.make_graph import build_graph
from deepfold.utils.model import load_model_checkpoint

sys.path.append('../')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config',
                                                 add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--test_file_name',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--namespace', default='', type=str, help='cc, mf, bp')
parser.add_argument('--model',
                    metavar='MODEL',
                    default='protgcn',
                    help='model architecture: (default: protgcn)')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256) per gpu')
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')


def main(args):
    args.gpu = 0
    # Dataset and DataLoader
    adj, multi_hot_vector, label_map, label_map_ivs = build_graph(
        data_path=args.data_path, namespace=args.namespace)
    nb_classes = len(label_map)
    test_dataset = GCNDataset(label_map,
                              root_path=args.data_path,
                              file_name=args.test_file_name)
    # dataloders
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    # model
    nodes = multi_hot_vector.cuda()
    adj = adj.cuda()
    model = MLP(1280,nb_classes)
    if args.resume is not None:
        if args.local_rank == 0:
            model_state, optimizer_state = load_model_checkpoint(args.resume)
            model.load_state_dict(model_state)

    # define loss function (criterion) and optimizer
    model = model.cuda()
    # run predict
    predictions, test_metrics = predict(model,
                                        test_loader,
                                        use_amp=args.amp,
                                        logger=logger)

    logger.info('Test metrics: %s' % (test_metrics))

    test_data_path = os.path.join(args.data_path, args.namespace,
                                  args.namespace + '_test_data.pkl')
    test_df = pd.read_pickle(test_data_path)

    preds, test_labels = predictions
    test_df['labels'] = list(test_labels)
    test_df['preds'] = list(preds)
    df_path = os.path.join(args.data_path, 'withoutGCN' + args.namespace + '_less_terms'+'_predictions.pkl')
    test_df.to_pickle(df_path)
    logger.info(f'Saving results to {df_path}')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    args, args_text = _parse_args()
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    task_name = 'ProtLM' + '_' + 'withoutGCN_' + args.namespace + '_less_terms_' + args.model
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True
    start_time = time.time()
    main(args)
