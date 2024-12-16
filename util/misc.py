# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from packaging import version
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

from util.box_ops import box_cxcywh_to_xyxy, box_iou
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class BasicLogger():
    def __init__(self):
        self.entries = {}   # dict containing loss logs
        self.stats_dict = {}
        self.stats_output = {}
        self.header_set = False

    def updateLosses(self, loss_dict, epoch, mode):
        """ Function to append loss dict to data logger
            Args:   loss_dict (dict): train / val loss dict
                    epoch (int):  current epoch
                    mode (str):   'train' / 'val' / 'test'
        """
        assert mode in ['train', 'val', 'test']
        if epoch not in self.entries:
            self.entries[epoch] = {mode: {}}
        self.entries[epoch][mode] = loss_dict

    def computePRData(self, outputs, queries_mask, labels_mask, mode):
        # Computes PR Curve for given data
        if 'PR' not in self.stats_dict[mode]:
            self.stats_dict[mode]['PR'] = {}
        thresh = np.linspace(0, 1, num=80)
        for t in thresh:
            t = round(t, 3)
            res = self.computeCFMetrics(outputs, queries_mask, labels_mask, thresh=t)
            if t not in self.stats_dict[mode]['PR']:
                self.stats_dict[mode]['PR'][t] = res
            else:
                self.stats_dict[mode]['PR'][t] = {k: res[k]+self.stats_dict[mode]['PR'][t][k] for k in res}

    def compute_mAPData(self, outputs, labels, queries_mask, labels_mask, iou_thresh=0.5, mode='val'):
        if 'mAP' not in self.stats_dict[mode]:
            self.stats_dict[mode]['mAP'] = {}
        if iou_thresh not in self.stats_dict[mode]['mAP']:
            self.stats_dict[mode]['mAP'][iou_thresh] = {}

        conf_threshs = np.linspace(start=0.05, stop=0.95, num=60)
        for c_t in conf_threshs:
            c_t = round(c_t, 3)
            if c_t not in self.stats_dict[mode]['mAP'][iou_thresh]:
                self.stats_dict[mode]['mAP'][iou_thresh][c_t] = {'tp': 0, 'fp': 0}

            src_logits = outputs['pred_logits'].cpu().detach()     # only get logits, that have corresponding label
            conf_mask = torch.full(src_logits.shape, fill_value=False)
            conf_mask[src_logits>=c_t] = True       # mask for conf logits that are greater than thresh
            bb_filtered = outputs['pred_boxes'][conf_mask & labels_mask].cpu().detach()
            labels_filtered = labels[conf_mask & labels_mask][...,1:]
            fp_conf = src_logits[~conf_mask & labels_mask].numel()  # fp through conf: has corresp label but is below conf level
            res = self.computeIOU(bb_filtered, labels_filtered)

            fp_iou = res[res<iou_thresh].numel()
            tp = res.numel() - fp_iou
            self.stats_dict[mode]['mAP'][iou_thresh][c_t]['tp'] += tp
            self.stats_dict[mode]['mAP'][iou_thresh][c_t]['fp'] += (fp_iou + fp_conf)

    def compute_mAP50_95(self, outputs, labels, queries_mask, labels_mask, mode='val'):
        threshs = np.arange(start=0.5, stop=1, step=0.05)
        for iou_thresh in threshs:
            self.compute_mAPData(outputs, labels, queries_mask, labels_mask, iou_thresh=iou_thresh, mode=mode)
        

    def computeIOU(self, bb_pred, bb_label):
        bb_pred = box_cxcywh_to_xyxy(bb_pred)
        bb_label = box_cxcywh_to_xyxy(bb_label)
        iou = box_iou(bb_pred, bb_label)[0]
        return torch.diag(iou)

    def computeCFMetrics(self, outputs, queries_mask, labels_mask, thresh = 0.5):
        # Computes TP / FP / TN / FN for given threshold
        src_logits = outputs['pred_logits'].cpu().detach()
        targets = torch.zeros_like(src_logits) 
        targets[labels_mask] = 1.0 # target mask to find labels idx where objectness = 1

        p = int(targets.sum().item())
        n = int(targets[queries_mask].numel() - targets.sum().item())

        f = src_logits[labels_mask].flatten()
        tp = f[f>thresh].numel()
        fn = f.numel()-tp

        f = src_logits[queries_mask & ~labels_mask].flatten()
        fp = f[f>thresh].numel()
        tn = f.numel()-fp

        return {"p": p, "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def computeStats(self, outputs, labels, queries_mask, labels_mask, mode='val'):
        assert mode in ['train', 'val', 'test']
        if mode not in self.stats_dict:
            self.stats_dict[mode] = {}
        self.computePRData(outputs, queries_mask, labels_mask, mode=mode)
        self.fcompute_mAP50_95(outputs, labels, queries_mask, labels_mask, mode='val')
        # self.compute_mAPData(outputs, labels, queries_mask, labels_mask, iou_thresh=0.5, mode='val')

    def resetStats(self):
        self.stats_dict = {}

    def print_mAP50(self, mode="val"):
        result = []
        target_dict = self.stats_dict[mode]['mAP'][0.5]
        for ct in target_dict:
            tp = target_dict[ct]['tp']
            fp = target_dict[ct]['fp']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            result.append(precision)
        result = round(sum(result)/len(result), 4)
        print("mAP@50: ", result)
        self.stats_output['mAP@50'] = result
        return result 

    def print_mAP50_95(self, mode="val"):
        result = []
        threshs = np.arange(0.5, 1, 0.05)
        for i in threshs:
            inter = []
            target_dict = self.stats_dict[mode]['mAP'][i]
            for ct in target_dict:
                tp = target_dict[ct]['tp']
                fp = target_dict[ct]['fp']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                inter.append(precision)
            map_i = sum(inter)/len(inter)
            result.append(map_i)
        result = round(sum(inter)/len(inter), 4)
        print("mAP@[.5:.95]: ", result)
        self.stats_output['mAP@[.5:.95]'] = result
        return result 

    def printCF(self, thresh=0.5, mode='val'):
        t_idx = np.argmin(np.abs(np.array([t for t in self.stats_dict[mode]['PR']]) - thresh))
        t = [t for t in self.stats_dict[mode]['PR']][t_idx]
        data = self.stats_dict[mode]['PR'][t]
        print("CF Objectness:", end="\t")
        for k in data:
            print(f"{k}: {data[k]}", end="\t\t")
        print()
        self.stats_output['CF Objectness'] = data

    def saveLossLogs(self, path):
        # Arg: path to training results folder
        with open(os.path.join(path, 'log_loss.txt'), 'w') as file:
            for epoch in self.entries:
                for mode in self.entries[epoch]:
                    results = self.entries[epoch][mode]
                    if not self.header_set:
                        self.header_set = True
                        header = "".ljust(20)+"".join([str(k).ljust(25) for k in results])
                        file.write(header + "\n")
                    line = f"Epoch {epoch} ({mode}):".ljust(20) + \
                        "".join([str(f"{k}: {round(v,3)}".ljust(25)) for k,v in results.items()])
                    file.write(line + "\n")

    def saveStatsLogs(self, path, epoch):
        # Arg: path to training results folder
        with open(os.path.join(path, 'log_stats.txt'), 'a') as file:
            start = f"Epoch {epoch}:".ljust(12)
            content = "".join([str(f"{k}: {v}".ljust(20)) for k,v in self.stats_output.items() if not isinstance(v,dict)])
            content_dict = "".join([str(f"{k}: {v}".ljust(20)) for d in [v for v in self.stats_output.values() if isinstance(v, dict)] for k, v in d.items()])
            file.write(start + content + content_dict + "\n")

    def plotPRCurve(self, path, mode='val'):
        pr_pairs = []
        for t in self.stats_dict[mode]['PR']:
            tp = self.stats_dict[mode]['PR'][t]['tp']
            fp = self.stats_dict[mode]['PR'][t]['fp']
            fn = self.stats_dict[mode]['PR'][t]['fn']
            precision = tp / (tp+fp) if (tp+fp) > 0 else 1
            recall = tp / (tp+fn) if (tp+fn) > 0 else 1
            pr_pairs.append([recall, precision, t])
        x,y,z = zip(*pr_pairs[::-1])

        z = np.array(z)
        norm = plt.Normalize(vmin=z.min(), vmax=z.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(z))

        fig, ax = plt.subplots()
        for i in range(len(x)-1):
            ax.plot(x[i:i+2], y[i:i+2], color=colors[i])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(z)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Threshold")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(path, 'PR_Curve.pdf'))

    def plotLoss(self, path):
        epochs = [k for k in self.entries]
        loss_train = [self.entries[k]['train']['loss_total'] for k in self.entries]
        loss_val = [self.entries[k]['val']['loss_total'] for k in self.entries]

        fig, ax = plt.subplots()
        ax.plot(epochs, loss_train, color='tab:blue', label='train')
        ax.plot(epochs, loss_val, color='tab:orange', label='val')
        ax.legend()
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")

        plt.savefig(os.path.join(path, 'Loss.pdf'))


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


