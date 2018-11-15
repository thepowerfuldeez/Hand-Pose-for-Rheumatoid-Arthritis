import datetime
import pathlib

import cv2
import torch as t
from matplotlib import pyplot as plt
from torch import nn as nn


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, pathlib.Path):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not path.exists():
        path.mkdir()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every args.lr_decay epochs"""
    # lr = 0.00005
    lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    # print("LR is " + str(lr)+ " at epoch "+ str(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def set_default_args(args):
    if not args.name:
        now = datetime.datetime.now()
        args.name = now.strftime("%Y-%m-%d-%H-%M")
    if not args.poses:
        args.poses = ["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']
    if not args.persons:
        args.persons = [0, 1, 2, 3, 4, 5, 6, 7]

    args.augment = not args.no_augment
    args.validate = not args.no_validate


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)


def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar'):
    expr_dir = opt.save_dir/opt.name
    t.save(state, str(expr_dir/filename))
    if is_best:
        t.save(state, str(expr_dir/'model_best.pth.tar'))


def load_checkpoint(path, model, optimizer):
    checkpoint = t.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def draw_pose(img, pose):
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)

    return img


def save_plt(array, name):
    plt.plot(array)
    plt.xlabel('epoch')
    plt.ylabel('name')
    plt.savefig(name + '.png')