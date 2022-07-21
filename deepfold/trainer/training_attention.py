import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.cuda.amp import autocast

from deepfold.core.metrics.custom_metrics import compute_roc
from deepfold.utils.metrics import AverageMeter
from deepfold.utils.model import reduce_tensor, save_checkpoint
from deepfold.utils.summary import update_summary


def get_train_step(model, optimizer, scaler, gradient_accumulation_steps,
                   use_amp):
    def _step(inputs, optimizer_step=True):
        # Runs the forward pass with autocasting.
        with autocast(enabled=use_amp):
            outputs = model(**inputs)
            loss = outputs[0]
            loss /= gradient_accumulation_steps
            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        scaler.scale(loss).backward()

        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        return reduced_loss

    return _step


def train(model,
          loader,
          optimizer,
          scaler,
          gradient_accumulation_steps,
          use_amp,
          epoch,
          logger,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_train_step(model, optimizer, scaler,
                          gradient_accumulation_steps, use_amp)

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(loader)
    end = time.time()
    for idx, batch in enumerate(loader):
        # Add batch to GPU
        batch = {key: val.cuda() for key, val in batch.items()}

        data_time = time.time() - end

        optimizer_step = ((idx + 1) % gradient_accumulation_steps) == 0
        loss = step(batch, optimizer_step)
        batch_size = batch['labels'].shape[0]

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)

        end = time.time()
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                log_name = 'Train-log'
                logger.info(
                    '{0}: [epoch:{1:>2d}] [{2:>2d}/{3}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'lr: {lr:>4.6f} '.format(log_name,
                                             epoch + 1,
                                             idx,
                                             steps_per_epoch,
                                             data_time=data_time_m,
                                             batch_time=batch_time_m,
                                             loss=losses_m,
                                             lr=learning_rate))

    return OrderedDict([('loss', losses_m.avg)])


def evaluate(model, loader, use_amp, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(loader)
    end = time.time()
    # Variables to gather full output
    true_labels, pred_labels = [], []
    for idx, batch in enumerate(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        labels = batch['labels']
        labels = labels.float()
        data_time = time.time() - end
        with torch.no_grad(), autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs[0]
            logits = outputs[1]

        torch.cuda.synchronize()

        preds = torch.sigmoid(logits)
        preds = preds.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        true_labels.append(labels)
        pred_labels.append(preds)

        batch_size = labels.shape[0]
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    # Flatten outputs
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    # avg_auc
    avg_auc = compute_roc(true_labels, pred_labels)
    metrics = OrderedDict([('loss', losses_m.avg), ('auc', avg_auc)])
    return metrics


def predict(model, loader, use_amp, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(loader)
    end = time.time()
    # Variables to gather full output
    true_labels, pred_labels = [], []
    for idx, batch in enumerate(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        labels = batch['labels']
        labels = labels.float()
        data_time = time.time() - end
        with torch.no_grad(), autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs[0]
            logits = outputs[1]

        torch.cuda.synchronize()

        preds = torch.sigmoid(logits)
        preds = preds.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        true_labels.append(labels)
        pred_labels.append(preds)

        batch_size = labels.shape[0]
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    # Flatten outputs
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    test_auc = compute_roc(true_labels, pred_labels)
    metrics = OrderedDict([('loss', losses_m.avg), ('auc', test_auc)])
    return (pred_labels, true_labels), metrics


def protlmpredict(model, loader, use_amp, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(loader)
    end = time.time()
    # Variables to gather full output
    true_labels, pred_labels = [], []
    for idx, batch in enumerate(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        labels = batch['labels']
        labels = labels.float()
        data_time = time.time() - end
        with torch.no_grad(), autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            logits = outputs[1] if isinstance(outputs,
                                              tuple) else outputs.logits

        preds = torch.sigmoid(logits)
        preds = preds.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        true_labels.append(labels)
        pred_labels.append(preds)

        batch_size = labels.shape[0]
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    # Flatten outputs
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    test_auc = compute_roc(true_labels, pred_labels)
    metrics = OrderedDict([('loss', losses_m.avg), ('auc', test_auc)])
    return (pred_labels, true_labels), metrics


def train_loop(model,
               optimizer,
               lr_scheduler,
               scaler,
               gradient_accumulation_steps,
               train_loader,
               val_loader,
               use_amp,
               logger,
               start_epoch=0,
               end_epoch=0,
               early_stopping_patience=-1,
               skip_training=False,
               skip_validation=True,
               save_checkpoints=True,
               output_dir='./',
               log_wandb=False,
               log_interval=10):
    is_best = True
    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    best_metric = np.inf
    logger.info('Evaluate validation set before start training')
    eval_metrics = evaluate(model, val_loader, use_amp, logger, log_interval)
    logger.info('Evaluation: %s' % (eval_metrics))
    logger.info(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        if not skip_training:
            train_metrics = train(model, train_loader, optimizer, scaler,
                                  gradient_accumulation_steps, use_amp, epoch,
                                  logger, log_interval)

            logger.info('[Epoch %d] training: %s' % (epoch + 1, train_metrics))

        if not skip_validation:
            eval_metrics = evaluate(model, val_loader, use_amp, logger,
                                    log_interval)

            logger.info('[Epoch %d] Evaluation: %s' %
                        (epoch + 1, eval_metrics))

        if eval_metrics['loss'] < best_metric:
            is_best = True
            best_metric = eval_metrics['loss']

        if log_wandb and (not torch.distributed.is_initialized()
                          or torch.distributed.get_rank() == 0):
            update_summary(epoch,
                           train_metrics,
                           eval_metrics,
                           os.path.join(output_dir, 'summary.csv'),
                           write_header=best_metric is None,
                           log_wandb=log_wandb)

        if save_checkpoints and (not torch.distributed.is_initialized()
                                 or torch.distributed.get_rank() == 0):
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': eval_metrics['loss'],
                'optimizer': optimizer.state_dict(),
            }
            logger.info('[*] Saving model epoch %d...' % (epoch + 1))
            save_checkpoint(checkpoint_state,
                            epoch,
                            is_best,
                            checkpoint_dir=output_dir)

        if early_stopping_patience > 0:
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            if epochs_since_improvement >= early_stopping_patience:
                break

        lr_scheduler.step()
