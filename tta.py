import time
import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def predict_pseudo_labels(probs, gt_labels=None):
    pred_labels = probs.argmax(dim=1)
    min_labels = probs.argmin(dim=1)
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100
    return min_labels, pred_labels, probs, accuracy


@torch.no_grad()
def eval_and_label_dataset(dataloader, model, sample_type, logger, acc_log, epoch, args, **kwargs):
    if epoch == -1:
        if sample_type == 'slow':
            logger.info("Source Only Slow Testing")
        else:
            logger.info("Source Only Fast Testing")
    else:
        if sample_type == 'slow':
            logger.info(f"Epoch[{epoch}]: Slow Testing")
        else:
            logger.info(f"Epoch[{epoch}]: Fast Testing")

    if sample_type == "slow":
        return_slow = True
    else:
        return_slow = False

    # make sure to switch to eval mode
    dataloader.dataset.eval()
    model.eval()
    if epoch >= 0:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.train()

    # run inference
    logits, gt_labels = [], []
    iterator = tqdm(dataloader)
    for slow_clips, fast_clips, labels in iterator:
        if sample_type=="slow":
            video = slow_clips.cuda()
        else:
            video = fast_clips.cuda()
        _, logits_cls = model(video, return_slow=return_slow, cls_only=True)
        logits.append(logits_cls)
        gt_labels.append(labels)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).cuda()

    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logger.info(f"Accuracy: {accuracy:.2f}")
    if sample_type == 'slow':
        acc_log.write(f"{accuracy},")
    else:
        acc_log.write(f"{accuracy}\n")
    acc_log.flush()


@torch.no_grad()
def eval_dataset_mixed(dataloader, model, logger, acc_log, epoch):
    if epoch == -1:
        logger.info("Source Only Testing")
        dataloader.dataset.eval()
        model.eval()
        slow_logits = list()
        fast_logits = list()
        mix_logits = list()
        gt_labels = list()
        for slow_clips, fast_clips, labels in tqdm(dataloader):
            _, slow_logits_cls = model(slow_clips.cuda(), return_slow=True, cls_only=True)
            slow_logits_cls = slow_logits_cls.cpu()
            _, fast_logits_cls = model(fast_clips.cuda(), return_slow=False, cls_only=True)
            fast_logits_cls = fast_logits_cls.cpu()
            
            mix_logits_cls = slow_logits_cls * 4 + fast_logits_cls # slow:fast 1:4
            slow_logits.append(slow_logits_cls)
            fast_logits.append(fast_logits_cls)
            mix_logits.append(mix_logits_cls)
            gt_labels.append(labels)
            
        slow_logits = torch.cat(slow_logits)
        fast_logits = torch.cat(fast_logits)
        mix_logits = torch.cat(mix_logits)
        gt_labels = torch.cat(gt_labels)
        slow_pred_labels = slow_logits.argmax(dim=1)
        fast_pred_labels = fast_logits.argmax(dim=1)
        mix_pred_labels = mix_logits.argmax(dim=1)
        slow_accuracy = (slow_pred_labels == gt_labels).float().mean() * 100
        fast_accuracy = (fast_pred_labels == gt_labels).float().mean() * 100
        mix_accuracy = (mix_pred_labels == gt_labels).float().mean() * 100
        logger.info(f"Slow acc:, {slow_accuracy:.2f}, Fast acc:, {fast_accuracy:.2f}, Mix acc:, {mix_accuracy:.2f}")
        acc_log.write(f"Slow acc:, {slow_accuracy}, Fast acc:, {fast_accuracy}, Mix acc:, {mix_accuracy}\n")
        acc_log.flush()
 
    else:
        dataloader.dataset.eval()
        model.eval()
        if epoch >= 0:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.train()
        gt_labels = list()
        slow_logits = list()
        fast_logits = list()
        mix_logits = list()
        for slow_clips, fast_clips, labels in tqdm(dataloader):
            _, slow_logits_cls = model(slow_clips.cuda(), return_slow=True, cls_only=True)
            slow_logits_cls = slow_logits_cls.cpu()
            _, fast_logits_cls = model(fast_clips.cuda(), return_slow=False, cls_only=True)
            fast_logits_cls = fast_logits_cls.cpu()
            
            mix_logits_cls = slow_logits_cls * 4 + fast_logits_cls
            slow_logits.append(slow_logits_cls)
            fast_logits.append(fast_logits_cls)
            mix_logits.append(mix_logits_cls)
            gt_labels.append(labels)

        slow_logits = torch.cat(slow_logits)
        fast_logits = torch.cat(fast_logits)
        mix_logits = torch.cat(mix_logits)
        gt_labels = torch.cat(gt_labels)
        slow_pred_labels = slow_logits.argmax(dim=1)
        fast_pred_labels = fast_logits.argmax(dim=1)
        mix_pred_labels = mix_logits.argmax(dim=1)
        slow_accuracy = (slow_pred_labels == gt_labels).float().mean() * 100
        fast_accuracy = (fast_pred_labels == gt_labels).float().mean() * 100
        mix_accuracy = (mix_pred_labels == gt_labels).float().mean() * 100
        logger.info(f"Slow acc:, {slow_accuracy:.2f}, Fast acc:, {fast_accuracy:.2f}, Mix acc:, {mix_accuracy:.2f}")
        acc_log.write(f"Slow acc:, {slow_accuracy}, Fast acc:, {fast_accuracy}, Mix acc:, {mix_accuracy}\n")
        acc_log.flush()
        

class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda() # labels: positive key indicators
    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    loss = F.cross_entropy(logits_ins, labels_ins)
    return loss


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def negative_loss(logits, labels, pure_random=False):
    logits = F.softmax(logits, dim=1)
    logits = 1 - logits
    onehot = torch.zeros_like(logits)
    for i in range(labels.shape[0]):
        if pure_random:
            chosen = random.randint(0, logits.shape[-1] - 1)
            onehot[i][chosen] = 1
        else:
            while True:
                chosen = random.randint(0, logits.shape[-1] - 1)
                if chosen != labels[i]:
                    break
            onehot[i][chosen] = 1
    return F.cross_entropy(onehot, logits)


@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy


def classification_loss(logits_w, logits_s, target_labels, ce_sup_type, cross_entropy, negative, pure_random=False):
    if ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels)
        accuracy = calculate_acc(logits_w, target_labels)
    elif ce_sup_type == "weak_strong":
        if cross_entropy:
            if negative:
                loss_cls1 = cross_entropy_loss(logits_s, target_labels)
                loss_cls2 = 0.1 * negative_loss(logits_s, target_labels)
                loss_cls = loss_cls1 + loss_cls2
            else:
                loss_cls = cross_entropy_loss(logits_s, target_labels)
        elif negative:
            loss_cls = 0.1 * negative_loss(logits_s, target_labels, pure_random)
        else:
            raise Exception("At least one classification loss!")
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError("CE supervision type not implemented.")
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))
    return loss_div


def diversification_loss(logits_w, logits_s, ce_sup_type):
    if ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)
    return loss_div

#训练一次周期
def train_epoch(train_loader, model, optimizer, logger, loss_log, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_slow = AverageMeter("CLS-Acc-Slow@1", ":6.2f")
    top1_fast = AverageMeter("CLS-Acc-Fast@1", ":6.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, loss_meter, top1_slow, top1_fast], prefix=f"Epoch: [{epoch}]")

    parallel = len(args.gpus) > 1

    # make sure to switch to train mode
    train_loader.dataset.train()
    model.train()
    end = time.time()
    #枚举train_loader中的data
    for i, data in enumerate(train_loader):
        # unpack and move data
        slow_video, fast_video, _ = data
        slow_video = slow_video.transpose(0, 1).contiguous()
        fast_video = fast_video.transpose(0, 1).contiguous()

        zero_tensor = torch.tensor([0.0]).to("cuda")

        slow_w, slow_q, slow_k = (slow_video[0], slow_video[1], slow_video[2])
        fast_w, fast_q, fast_k = (fast_video[0], fast_video[1], fast_video[2])

        # w and pseudo labels
        if slow_w.shape[1] == 1:
            inputs_w = torch.cat([slow_w, fast_w], dim=1)
        else:
            inputs_w = torch.cat([slow_w.unsqueeze(dim=1), fast_w.unsqueeze(dim=1)], dim=1)
        with torch.no_grad():
            feats_slow_w, logits_slow_w, feats_fast_w, logits_fast_w = model(inputs_w.cuda(), return_slow=None,
                                                                             cls_only=True, both=True, alignment=False)

            probs_slow = F.softmax(logits_slow_w, dim=1)
            min_labels_slow, pseudo_labels_slow, probs_slow, _ = predict_pseudo_labels(probs=probs_slow)
            probs_fast = F.softmax(logits_fast_w, dim=1)
            min_labels_fast, pseudo_labels_fast, probs_fast, _ = predict_pseudo_labels(probs=probs_fast)
            del probs_slow, probs_fast

        # q and k
        if slow_q.shape[1] == 1:
            inputs_q = torch.cat([slow_q, fast_q], dim=1)
        else:
            inputs_q = torch.cat([slow_q.unsqueeze(dim=1), fast_q.unsqueeze(dim=1)], dim=1)
        if slow_k.shape[1] == 1:
            inputs_k = torch.cat([slow_k, fast_k], dim=1)
        else:
            inputs_k = torch.cat([slow_k.unsqueeze(dim=1), fast_k.unsqueeze(dim=1)], dim=1)
        if args.alignment:
            feats_slow_q, logits_slow_q, feats_fast_q, logits_fast_q, k_slow, k_fast, l_slow, l_fast, means_slow, vars_slow, means_fast, vars_fast = model(inputs_q.cuda(), inputs_k.cuda(), return_slow=None, cls_only=False, both=True, alignment=True)
        else:
            feats_slow_q, logits_slow_q, feats_fast_q, logits_fast_q, k_slow, k_fast, l_slow, l_fast = model(inputs_q.cuda(), inputs_k.cuda(), return_slow=None, cls_only=False, both=True, alignment=False)

        # update key features and corresponding pseudo labels
        if parallel:
            model.module.update_memory(k_slow, pseudo_labels_slow)
            model.module.update_memory(k_fast, pseudo_labels_fast)
        else:
            model.update_memory(k_slow, pseudo_labels_slow)
            model.update_memory(k_fast, pseudo_labels_fast)

        # contrastive learning
        mem_labels = model.module.mem_labels if parallel else model.mem_labels
        loss_nce_slow = instance_loss(logits_ins=l_slow, pseudo_labels=pseudo_labels_slow,
                                      mem_labels=mem_labels, contrast_type=args.contrast_type)
        loss_nce_fast = instance_loss(logits_ins=l_fast, pseudo_labels=pseudo_labels_fast,
                                      mem_labels=mem_labels, contrast_type=args.contrast_type)
        loss_nce = 0.5 * loss_nce_slow + 0.5 * loss_nce_fast

        # classification
        loss_cls_slow, accuracy_slow = classification_loss(logits_slow_w, logits_slow_q, pseudo_labels_slow,
                                                           ce_sup_type=args.ce_sup_type,
                                                           cross_entropy=args.cross_entropy,
                                                           negative=args.negative,
                                                           pure_random=args.pure_random)
        loss_cls_fast, accuracy_fast = classification_loss(logits_fast_w, logits_fast_q, pseudo_labels_fast,
                                                           ce_sup_type=args.ce_sup_type,
                                                           cross_entropy=args.cross_entropy,
                                                           negative=args.negative,
                                                           pure_random=args.pure_random)
        top1_slow.update(accuracy_slow.item(), len(logits_slow_w))
        top1_fast.update(accuracy_fast.item(), len(logits_fast_w))
        loss_cls = 0.5 * loss_cls_slow + 0.5 * loss_cls_fast

        # diversification
        loss_div_slow = diversification_loss(logits_slow_w, logits_slow_q, args.ce_sup_type) if args.eta > 0 else zero_tensor
        loss_div_fast = diversification_loss(logits_fast_w, logits_fast_q, args.ce_sup_type) if args.eta > 0 else zero_tensor
        loss_div = 0.5 * loss_div_slow + 0.5 * loss_div_fast

        # alignment
        if args.alignment:
            if args.alignment_half:
                loss_alm = list()
                for mean_slow, var_slow, mean_fast, var_fast in zip(means_slow, vars_slow, means_fast, vars_fast):
                    num_gpus = len(args.gpus)
                    mean_slow = mean_slow.detach().view(num_gpus, -1).mean(dim=0, keepdim=False)
                    var_slow = var_slow.detach().view(num_gpus, -1).mean(dim=0, keepdim=False)
                    mean_fast = mean_fast.view(num_gpus, -1).mean(dim=0, keepdim=False)
                    var_fast = var_fast.view(num_gpus, -1).mean(dim=0, keepdim=False)
                    mean_sub = mean_slow.cuda() - mean_fast.cuda()
                    mean_sub = mean_sub.norm(1)  # L1 norm
                    var_sub = var_slow.cuda() - var_fast.cuda()
                    var_sub = var_sub.norm(1)
                    l = mean_sub + var_sub
                    loss_alm.append(l)
                loss_alm = torch.stack(loss_alm)
                loss_alm = loss_alm.sum()
            else:
                loss_alm = list()
                for mean_slow, var_slow, mean_fast, var_fast in zip(means_slow, vars_slow, means_fast, vars_fast):
                    mean_sub = mean_slow.cuda() - mean_fast.cuda()
                    mean_sub = mean_sub.norm(1)  # L1 norm
                    var_sub = var_slow.cuda() - var_fast.cuda()
                    var_sub = var_sub.norm(1)
                    l = mean_sub + var_sub
                    loss_alm.append(l)
                loss_alm = torch.stack(loss_alm)
                loss_alm = loss_alm.sum()

        if args.alignment:
            loss = args.alpha * loss_cls + args.beta * loss_nce + args.eta * loss_div + args.alm * loss_alm
        else:
            loss = args.alpha * loss_cls + args.beta * loss_nce + args.eta * loss_div

        if i % args.report_frequence == 0:
            if args.alignment:
                loss_log.write(f"{epoch},{i},{loss.item()},{loss_cls.item()},{loss_nce.item()},{loss_div.item()},{loss_alm.item()}\n")
                loss_log.flush()
            else:
                loss_log.write(f"{epoch},{i},{loss.item()},{loss_cls.item()},{loss_nce.item()},{loss_div.item()}\n")
                loss_log.flush()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_meter.update(loss.item())

        with torch.no_grad():
            slow_feats_w, slow_logits_w = model.momentum_model(slow_w.cuda(), return_slow=True, return_feats=True)
        with torch.no_grad():
            fast_feats_w, fast_logits_w = model.momentum_model(fast_w.cuda(), return_slow=False, return_feats=True)
            

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.report_frequence == 0:
            progress.display(i, logger)
