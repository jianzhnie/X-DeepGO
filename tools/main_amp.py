import argparse
import logging
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import yaml
from torch.utils.data import DataLoader

from deepfold.data.esm_dataset import ESMDataset
from deepfold.models.esm_model import ESMTransformer
from deepfold.scheduler.lr_scheduler import LinearLRScheduler
from deepfold.trainer.training_amp import train_loop

sys.path.append('../')

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp
except ImportError:
    raise ImportError(
        'Please install apex from https://www.github.com/nvidia/apex to run this example.'
    )

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

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
parser.add_argument('--model',
                    metavar='MODEL',
                    default='esm',
                    help='model architecture: (default: esm)')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
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
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--end-lr',
                    '--minimum learning-rate',
                    default=1e-8,
                    type=float,
                    metavar='END-LR',
                    help='initial learning rate')
parser.add_argument(
    '--lr-schedule',
    default='step',
    type=str,
    metavar='SCHEDULE',
    choices=['step', 'linear', 'cosine', 'exponential'],
    help='Type of LR schedule: {}, {}, {} , {}'.format('step', 'linear',
                                                       'cosine',
                                                       'exponential'),
)
parser.add_argument('--warmup',
                    default=0,
                    type=int,
                    metavar='E',
                    help='number of warmup epochs')
parser.add_argument('--optimizer',
                    default='sgd',
                    type=str,
                    choices=('sgd', 'rmsprop', 'adamw'))
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp',
                    action='store_true',
                    default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp',
                    action='store_true',
                    default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument(
    '--early-stopping-patience',
    default=-1,
    type=int,
    metavar='N',
    help='early stopping after N epochs without improving',
)
parser.add_argument(
    '--gradient_accumulation_steps',
    default=1,
    type=int,
    metavar='N',
    help='=To run gradient descent after N steps',
)
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--training-only',
                    action='store_true',
                    help='do not evaluate')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument(
    '--static-loss-scale',
    type=float,
    default=1,
    help='Static loss scale',
)
parser.add_argument(
    '--dynamic-loss-scale',
    action='store_true',
    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
    '--static-loss-scale.',
)
parser.add_argument(
    '--no-checkpoints',
    action='store_false',
    dest='save_checkpoints',
    help='do not store any checkpoints, useful for benchmarking',
)
parser.add_argument('--checkpoint-filename',
                    default='checkpoint.pth.tar',
                    type=str)
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')
parser.add_argument('--log-wandb',
                    action='store_true',
                    help='while to use wandb log systerm')


def main(args):
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                'Metrics not being logged to wandb, try `pip install wandb`')

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = os.getenv('LOCAL_RANK', 0)

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.gpu = 0
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.gpu = args.local_rank
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        logger.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size))
    else:
        logger.info('Training with a single process on %s .' % args.device)

    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'
    assert args.rank >= 0

    # get data loaders
    # Dataset and DataLoader
    train_dataset = ESMDataset(data_path=args.data_path,
                               split='train',
                               model_dir='esm1b_t33_650M_UR50S')

    test_dataset = ESMDataset(data_path=args.data_path,
                              split='test',
                              model_dir='esm1b_t33_650M_UR50S')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    # dataloders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        collate_fn=train_dataset.collate_fn,
        sampler=test_sampler,
        pin_memory=True,
    )

    # model
    num_labels = train_dataset.num_classes
    model = ESMTransformer(model_dir='esm1b_t33_650M_UR50S',
                           pool_mode='cls',
                           num_labels=num_labels)
    # define loss function (criterion) and optimizer
    # optimizer and lr_policy
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_policy = LinearLRScheduler(optimizer=optimizer,
                                  base_lr=args.lr,
                                  warmup_length=args.warmup,
                                  epochs=args.epochs,
                                  logger=logger)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model,
                                      optimizer,
                                      opt_level=args.opt_level,
                                      loss_scale=args.loss_scale)
    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_policy is not None and start_epoch > 0:
        lr_policy.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(args.epochs))

    train_loop(
        model,
        optimizer,
        lr_policy,
        train_loader,
        test_loader,
        device=args.device,
        logger=logger,
        start_epoch=start_epoch,
        end_epoch=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.output_dir,
        checkpoint_filename=args.checkpoint_filename,
    )
    print('Experiment ended')


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

    task_name = 'ProtLM' + '_' + args.model
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
