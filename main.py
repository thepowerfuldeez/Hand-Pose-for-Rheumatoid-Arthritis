import pathlib

import warnings
import torch as t
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import time
import argparse
import os

from MSRADataset import MSRADataset, _unnormalize_joints
from REN import REN
from loss import Modified_SmoothL1Loss
from utils import adjust_learning_rate, set_default_args, weights_init, save_checkpoint, load_checkpoint, \
    save_plt, mkdirs

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Region Ensemble Network')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=40, help='number of epochs')
parser.add_argument('--test', action='store_true', help='only test without training')
parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='decay lr by 10 after _ epoches')
parser.add_argument('--input_size', type=int, default=96, help='decay lr by 10 after _ epoches')
parser.add_argument('--num_joints', type=int, default=42, help='decay lr by 10 after _ epoches')
parser.add_argument('--no_augment', action='store_true', help='dont augment data?')
parser.add_argument('--no_validate', action='store_true', help='dont validate data when training?')
parser.add_argument('--augment_probability', type=float, default=1.0, help='augment probability')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--poses', type=str, default=None, nargs='+', help='poses to train on')
parser.add_argument('--persons', type=str, default=None, nargs='+', help='persons to train on')
parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')
parser.add_argument('--print_interval', type=int, default=500, help='print interval')
parser.add_argument('--save_dir', type=str, default="experiments/", help='path/to/save_dir')
parser.add_argument('--name', type=str, default=None,
                    help='name of the experiment. It decides where to store samples and models. if none, '
                         'it will be saved as the date and time')
parser.add_argument('--finetune', action='store_true', help='use a pretrained checkpoint')


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    expr_dir = opt.save_dir / opt.name
    mkdirs(expr_dir)
    file_name = expr_dir / 'opt.txt'
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def main(args):
    set_default_args(args)
    model = REN(args)

    model.float()
    model.cuda()
    model.apply(weights_init)
    cudnn.benchmark = True
    criterion = Modified_SmoothL1Loss().cuda()

    train_dataset = MSRADataset(training=True, augment=args.augment, args=args)
    test_dataset = MSRADataset(training=False, augment=False, args=args)

    train_loader = t.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=True,
        num_workers=0, pin_memory=False)

    val_loader = t.utils.data.DataLoader(
        test_dataset, batch_size=args.batchSize, shuffle=True,
        num_workers=0, pin_memory=False)

    optimizer = t.optim.Adam(model.parameters(), args.lr,
                             # momentum=args.momentum,
                             weight_decay=args.weight_decay)

    current_epoch = 0
    if args.checkpoint:
        model, optimizer, current_epoch = load_checkpoint(args.checkpoint, model, optimizer)
        if args.finetune:
            current_epoch = 0

    if args.test:
        test(model, args)
        return

    train_loss = []
    val_loss = []
    best = False

    print_options(args)
    expr_dir = args.save_dir / args.name

    for epoch in range(current_epoch, args.epoch):

        optimizer = adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch, args)
        train_loss = train_loss + loss_train
        if args.validate:
            # evaluate on validation set
            loss_val = validate(val_loader, model, criterion, args)
            val_loss = val_loss + loss_val

        state = {
            'epoch': epoch,
            'arch': type(model).__name__,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if not (expr_dir / 'model_best.pth.tar').exists():
            save_checkpoint(state, True, args)

        if args.validate and epoch > 1:
            best = (loss_val < min(val_loss[:len(val_loss) - 1]))
            if best:
                print("saving best performing checkpoint on val")
                save_checkpoint(state, True, args)

        save_checkpoint(state, False, args)
    #

    expr_dir = args.save_dir / args.name
    np.savetxt(str(expr_dir / "train_loss.out"), train_loss, fmt='%f')
    save_plt(train_loss, "train_loss")
    np.savetxt(str(expr_dir / "val_loss.out"), val_loss, fmt='%f')
    save_plt(val_loss, "val_loss")


def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()
    loss_train = []
    for i, (input, target) in enumerate(train_loader):
        stime = time.time()
        # measure data loading time
        target = target.float()
        target = target.cuda(non_blocking=False)
        input = input.float()
        input = input.cuda()
        # compute output
        output = model(input)

        loss = criterion(output, target)
        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        loss_train.append(loss.data.item())
        optimizer.step()
        # measure elapsed time
        if i % args.print_interval == 0:
            TT = time.time() - stime
            print('epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Time: {time:.2f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item(), time=TT))

    return [np.mean(loss_train)]


def validate(val_loader, model, criterion, args):
    # switch to evaluate mode
    model.eval()

    loss_val = []
    with t.no_grad():
        expr_dir = os.path.join(args.save_dir, args.name)

        for i, (input, target) in enumerate(val_loader):
            target = target.float()
            target = target.cuda(non_blocking=False)
            # compute output
            input = input.float()
            input = input.cuda()
            output = model(input)
            loss = criterion(output, target)

            if i % args.print_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                    i, len(val_loader), loss=loss))
            loss_val.append(loss.data.item())

    return [np.mean(loss_val)]


def test(model, args):
    # switch to evaluate mode
    model.eval()
    test_dataset = MSRADataset(training=False, augment=False, args=args)
    errors = []
    MAE_criterion = nn.L1Loss()
    with t.no_grad():
        expr_dir = os.path.join(args.save_dir, args.name)

        input_size = args.input_size
        for i, (input, target) in enumerate(test_dataset):
            target = target.float()
            target = target.numpy().reshape(21, 2)
            tmp = np.zeros((21, 3))
            for j in range(len(target)):
                tmp[j, :2] = target[j]
            # compute output
            input = input.float()
            input = input.cuda()
            input = input.unsqueeze(0)
            output = model(input)
            output = output.cpu().numpy().reshape(21, 2)
            tmp1 = np.zeros((21, 3))
            for j in range(len(output)):
                tmp1[j, :2] = output[j]
            center = test_dataset.get_center(i)
            # errors.append(compute_distance_error(_unnormalize_joints(tmp1,center,input_size),
            # _unnormalize_joints(tmp,center,input_size)).item())
            output = t.from_numpy(_unnormalize_joints(tmp1, center, input_size))
            target = t.from_numpy(_unnormalize_joints(tmp, center, input_size))
            MAE_loss = MAE_criterion(output, target)

            errors.append(MAE_loss.item())

            if i % args.print_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                    i, len(test_dataset), loss=errors[-1]))

        errors = np.mean(errors)
        print(errors)
        if "model_best" in args.checkpoint:
            np.savetxt(os.path.join(expr_dir, "average_MAE_model_best_" + args.poses[0]), np.asarray([errors]),
                       fmt='%f')
        else:
            np.savetxt(os.path.join(expr_dir, "average_MAE_checkpoint" + args.poses[0]), np.asarray([errors]), fmt='%f')


if __name__ == '__main__':
    args = parser.parse_args()
    args.save_dir = pathlib.Path(args.save_dir)
    main(args)
