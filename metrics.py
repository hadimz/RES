
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def compute_iou(x, y):
    intersection = np.bitwise_and(x, y)
    union = np.bitwise_or(x, y)

    iou = np.sum(intersection) / np.sum(union)

    return iou

def compute_exp_score(x, y):
    N =  np.sum(y!=0)
    epsilon = 1e-6
    tp = np.sum( x * (y>0))
    tn = np.sum((1-x) * (y<0))
    fp = np.sum( x * (y<0))
    fn = np.sum((1-x) * (y>0))

    exp_precision = tp / (tp + fp + epsilon)
    exp_recall = tp / (tp + fn + epsilon)
    exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall + epsilon)

    return exp_precision, exp_recall, exp_f1

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

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

