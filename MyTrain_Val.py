# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr,dice_coefficient
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()

            preds = model(images)
            loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts)
            loss_final = structure_loss(preds[3], gts)

            loss = loss_init + loss_final

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                res = preds[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch_mae
    global best_dice, best_epoch_dice
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        dice_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt_dice = torch.tensor(gt / 255).cuda()
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[3], size=gt.shape, mode='bilinear', align_corners=False)
            val_output = res.sigmoid().data.squeeze()
            val_output[val_output>=0.5] = 1
            val_output[val_output<0.5] = 0
            dice_sum += dice_coefficient(gt_dice,val_output)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        dice = dice_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        if epoch == 1:
            best_mae = mae
            best_dice = dice
            best_epoch_mae = epoch
            best_epoch_dice = epoch
        elif epoch == 0:
            best_mae = mae
            best_dice = dice
            best_epoch_mae = epoch
            best_epoch_dice = epoch
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch_mae = epoch
                torch.save(model.state_dict(), save_path + 'Net_mae_best.pth')
                print('Save state_dict successfully! Best mae epoch:{}.'.format(epoch))

            if dice > best_dice:
                best_dice = dice
                best_epoch_dice = epoch
                torch.save(model.state_dict(), save_path + 'Net_dice_best.pth')
                print('Save state_dict successfully! Best dice epoch:{}.'.format(epoch))
        print('Epoch: {}, MAE: {}, DICE: {}, bestMAE: {}, bestEpoch_mae: {}, bestDICE: {}, bestEpoch_dice: {}.'.format(epoch, mae, dice, best_mae, best_epoch_mae, best_dice, best_epoch_dice))
        logging.info(
            '[Val Info]:Epoch: {}, MAE: {}, DICE: {}, bestMAE: {}, bestEpoch_mae: {}, bestDICE: {}, bestEpoch_dice: {}.'.format(epoch, mae, dice, best_mae, best_epoch_mae, best_dice, best_epoch_dice))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default='./Net_dice_best.pth', help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./data/train/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./data/test/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./Save/test/',
                        help='the path to save model and log')
    parser.add_argument('--mode', type=str, default="Test", help='choose Train or Test') 
    opt = parser.parse_args()

    cudnn.benchmark = True

    # build the model
    model = Network(channel=32).cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'crop_images/',
                              gt_root=opt.train_root + 'crop_masks/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=0)
    val_loader = test_dataset(image_root=opt.val_root + 'crop_images/',
                              gt_root=opt.val_root + 'crop_masks/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    if opt.mode == "Train":
    # logging
        logging.basicConfig(filename=save_path + 'log.log',
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        logging.info("Network-Train")
        logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                    'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                            opt.decay_rate, opt.load, save_path, opt.decay_epoch))

        step = 0
        writer = SummaryWriter(save_path + 'summary')
        best_mae = 1
        best_epoch = 0
        epoch = 1
        for epoch in range(1, opt.epoch + 1):
            if epoch == 1:    
                print("Start train...")
            cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
            train(train_loader, model, optimizer, epoch, save_path, writer)
            val(val_loader, model, epoch, save_path, writer)
    elif opt.mode == "Test":
        epoch = 0
        writer = SummaryWriter(save_path + 'summary')
        val(val_loader, model, epoch, save_path, writer)