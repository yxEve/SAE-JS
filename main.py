# -*- coding:utf-8 -*-
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import os
import time
import random
import torchvision.utils as vutils
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from critic import BasicCritic
from generator import GeneratorResnet
from steganalysis.UCNet_JPEG import Net as Net1
from steganalysis.UCNet_Spatial import Net as Net2
from jpeg.IDiffJPEG import IDCT, YCbCr_Space
import utils
import pytorch_ssim
from jpeg.compression import rgb_to_ycbcr_jpeg


def main():

    parser = argparse.ArgumentParser(description='Training of our nets')

    parser.add_argument('--train-data-dir', required=True, type=str,
                                help='The directory where the data for training is stored.')
    parser.add_argument('--valid-data-dir', required=True, type=str,
                                help='The directory where the data for validation is stored.')
    parser.add_argument('--run-folder', type=str, required=True,
                                help='The experiment folder where results are logged.')
    parser.add_argument('--title', type=str, required=True, help='The experiment name.')

    parser.add_argument('--size', default=256, type=int,
                        help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--data-depth', default=1, type=int, help='The depth of the message.')

    parser.add_argument('--batch-size', type=int, help='The batch size.', default=4)
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs.')

    parser.add_argument('--gray', action='store_true', default=False, help='Use gray-scale images.')
    parser.add_argument('--hidden-size', type=int, default=32, help='Hidden channels in networks.')
    parser.add_argument('--tensorboard', action='store_true', help='Use to switch on Tensorboard logging.')
    parser.add_argument('--seed', type=int, default=20, help='Random seed.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of gpu used (default: 0).')
    parser.add_argument('--use-vgg', action='store_true', default=True, help='Use VGG loss.')


    args = parser.parse_args()
    use_discriminator = True

    log_dir = os.path.join(args.run_folder, time.strftime("%Y-%m-%d--%H-%M-%S-") + args.title)
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(checkpoints_dir)

    train_csv_file = os.path.join(log_dir, args.title + '_train.csv')
    valid_csv_file = os.path.join(log_dir, args.title + '_valid.csv')

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    # Load Datasets
    print('---> Loading Datasets...')
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop((args.size, args.size)),
    ])
    train_dataset = utils.Mydataset(args.train_data_dir, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)

    valid_transform = transforms.Compose([
        # transforms.CenterCrop((args.size, args.size))
    ])
    valid_dataset = utils.Mydataset(args.valid_data_dir, valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False, **kwargs)

    message_path = '/home/data/qr_dataset_real256'
    message = utils.Mymessage(message_path)
    message_loader = DataLoader(message, batch_size=args.batch_size, drop_last=False, shuffle=True, **kwargs)

    # Load Models
    print('---> Constructing Network Architectures...')
    color_band = 1 if args.gray else 3
    encoder = Encoder(args.data_depth, args.hidden_size, color_band)
    decoder = Decoder(args.data_depth, args.hidden_size, color_band)
    discriminator = BasicCritic(args.hidden_size)
    netG = GeneratorResnet(eps=0.59)
    netT1 = Net1()
    netT2 = Net2()
    get_spatial = IDCT(height=256, width=256, differentiable=False, quality=75)
    get_ycbcr = YCbCr_Space(height=256, width=256, differentiable=True, quality=75)
    rgb_to_ycbcr = rgb_to_ycbcr_jpeg()


    # VGG for perceptual loss
    print('---> Constructing VGG-16 for Perceptual Loss...')
    vgg = utils.VGGLoss(3, 1, False)

    # Define Loss
    print('---> Defining Loss...')
    optimizer_coders = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer_coders, step_size=10, gamma=0.1)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=0)
    optimizer_netG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=0)
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    def round_func_BPDA(x):
        # This is equivalent to replacing round function (non-differentiable) with
        # an identity function (differentiable) only when backward.
        # forward_value = torch.round(input)
        # out = input.clone()
        # out.data = forward_value.data
        return torch.round(x) + (x - torch.round(x))**3

    def norm(channel):
        max_val = torch.max(channel)
        min_val = torch.min(channel)
        channel = (channel - min_val) / (max_val - min_val + 1e-06)
        return channel

    # 纹理损失权重
    def M_texture(I):
        paddings = (1, 1, 1, 1)
        I0 = F.pad(I, paddings, "reflect")
        I0 = F.avg_pool2d(I0, kernel_size=(3, 3), stride=(1, 1), padding=0)
        I0 = I0 - I / 9
        Itexture = torch.sqrt(torch.square(I - I0))
        A = torch.ones_like(Itexture)
        return A - norm(Itexture)

    def M_color(I):
        ycbcr = rgb_to_ycbcr(I)
        y, cb, cr = torch.split(ycbcr, 1, dim=3)
        cr = torch.squeeze(cr, -1)
        cr = 1 - torch.exp(-(norm(cr)))
        cr = torch.unsqueeze(cr, -1)
        y = torch.ones_like(y)
        cb = torch.ones_like(cb)
        A = torch.cat((y, cb, cr), 3).permute(0, 3, 1, 2)
        return A


    # Use GPU
    if args.cuda:
        print('---> Loading into GPU memory...')
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        netG.cuda()
        netT1.cuda()
        netT2.cuda()
        mse_loss.cuda()
        bce_loss.cuda()
        vgg.cuda()
        get_spatial.cuda()
        get_ycbcr.cuda()
        rgb_to_ycbcr.cuda()

        steganalyzer_para1 = '/home/UCNet_Steganalysis-main/UCNet_JPEG/0.4-0.02-params-lr=75.pt'
        all_state1 = torch.load(steganalyzer_para1)
        original_state1 = all_state1['original_state']
        netT1.load_state_dict(original_state1)
        netT1.eval()

        steganalyzer_para2 = '/home/UCNet_Steganalysis-main/UCNet_Spatial/params-lr=0.01.pt'
        all_state2 = torch.load(steganalyzer_para2)
        original_state2 = all_state2['original_state']
        netT2.load_state_dict(original_state2)
        netT2.eval()

        for p in netT1.parameters():
            p.requires_grad = False

        for p in netT2.parameters():
            p.requires_grad = False

    start_epoch = 0
    iteration = 0

    metric_names = ['a_loss', 'adv_loss', 'mse_loss', 'ssim_loss', 'vgg_loss', 'decoder_loss', 'loss',
                    'bit_err', 'decode_accuracy', 'psnr', 'ssim']
    metrics = {m: 0 for m in metric_names}

    tic = time.time()
    for e in range(start_epoch, args.epochs):
        print('---> Epoch %d starts training...' % e)
        epoch_start_time = time.time()
        # ------ train ------
        encoder.train()
        decoder.train()
        discriminator.train()
        netG.train()
        i = 0  # batch idx
        train_iter = iter(train_loader)
        while i < len(train_loader):
            # ---------------- Train the discriminator -----------------------------
            if use_discriminator:
                Diters = 5
                j = 0
                while j < Diters and i < len(train_loader):
                    for p in discriminator.parameters():  # reset requires_grad
                        p.requires_grad = True
                    image_in = next(train_iter).cuda()

                    batch_size, _, h, w = image_in.size()
                    message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
                    # message_in = next(iter(message_loader))

                    if args.cuda:
                        message_in = message_in.cuda()
                        image_in = image_in.cuda()

                    optimizer_discriminator.zero_grad()
                    stego = encoder(image_in, message_in)
                    adv_stego, _, _, _ = netG(stego)
                    stego1 = stego.permute(0, 2, 3, 1)
                    adv_stego1 = round_func_BPDA(adv_stego).permute(0, 2, 3, 1)
                    image_in1 = image_in.permute(0, 2, 3, 1)
                    spa_stego = get_spatial(stego1).cuda()  # spatial domain
                    spa_adv_stego = get_spatial(adv_stego1).cuda()
                    spa_cover = get_spatial(image_in1).cuda()

                    d_on_spa_cover = discriminator(spa_cover)
                    d_on_spa_adv_stego = discriminator(spa_adv_stego)

                    d_loss = d_on_spa_adv_stego.mean() - d_on_spa_cover.mean()
                    d_loss.backward()
                    optimizer_discriminator.step()

                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    j += 1
                    i += 1

            if i == len(train_loader):
                break

            # -------------- Train the generator ---------------------
            for p in discriminator.parameters():
                p.requires_grad = False

            image_in = next(train_iter).cuda()
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            # message_in = next(iter(message_loader))

            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()

            optimizer_netG.zero_grad()
            stego = encoder(image_in, message_in)  # coefficients of stego
            adv_stego, adv_inf, adv_0, adv_00 = netG(stego)
            extract_message = decoder(adv_stego)
            adv_stego1 = round_func_BPDA(adv_stego).permute(0, 2, 3, 1)
            spa_adv_stego = get_spatial(adv_stego1).cuda()
            spa_adv_stego_round = round_func_BPDA(spa_adv_stego).clamp(0, 255)
            ycbcr_adv_stego = get_ycbcr(adv_stego1).permute(0, 3, 1, 2).cuda()
            output1 = netT1(ycbcr_adv_stego)
            output2 = netT2(spa_adv_stego_round)

            curr_adv_label1 = output1.max(1, keepdim=True)[1]  # 找出最大的下标
            targ_adv_label1 = torch.zeros_like(curr_adv_label1).cuda()  #0是cover，1是stego
            curr_adv_pred1 = output1.gather(1, curr_adv_label1)
            targ_adv_pred1 = output1.gather(1, targ_adv_label1)
            attack_loss1 = torch.mean(curr_adv_pred1 - targ_adv_pred1)

            curr_adv_label2 = output2.max(1, keepdim=True)[1]  # 找出最大的下标
            targ_adv_label2 = torch.zeros_like(curr_adv_label2).cuda()  # 0是cover，1是stego
            curr_adv_pred2 = output2.gather(1, curr_adv_label2)
            targ_adv_pred2 = output2.gather(1, targ_adv_label2)
            attack_loss2 = torch.mean(curr_adv_pred2 - targ_adv_pred2)

            attack_loss = (attack_loss1 + attack_loss2) / 2

            spa_loss = torch.norm(adv_0, 1)
            bi_adv_00 = torch.where(adv_00 < 0.95, torch.zeros_like(adv_00), torch.ones_like(adv_00))
            qua_loss = torch.sum((bi_adv_00 - adv_00) ** 2)
            a_loss = attack_loss + 0.0001 * spa_loss + 0.0001 * qua_loss

            a_loss.backward()
            optimizer_netG.step()

            i += 1

            if i == len(train_loader):
                break

            # -------------- Train the (encoder-decoder) ---------------------
            for p in discriminator.parameters():
                p.requires_grad = False

            image_in = next(train_iter).cuda()
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            # message_in = next(iter(message_loader))

            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()

            optimizer_coders.zero_grad()
            stego = encoder(image_in, message_in)
            adv_stego, adv_inf, adv_0, adv_00 = netG(stego)
            extract_message = decoder(adv_stego)
            adv_stego1 = round_func_BPDA(adv_stego).permute(0, 2, 3, 1)
            spa_adv_stego = get_spatial(adv_stego1).cuda()

            image_in1 = image_in.permute(0, 2, 3, 1)
            stego1 = stego.permute(0, 2, 3, 1)
            spa_stego = get_spatial(stego1).cuda()  # spatial domain
            spa_cover = round_func_BPDA(get_spatial(image_in1)).cuda()

            g_on_adv_stego = discriminator(spa_adv_stego)

            M = (M_texture(spa_cover) + M_color(spa_cover)) / 2
            diff = norm(spa_cover) - norm(spa_adv_stego)

            eye_loss = torch.mean(torch.abs(M * diff))

            g_adv_loss = - g_on_adv_stego.mean()
            g_mse_loss = mse_loss(spa_adv_stego, spa_cover) + eye_loss
            g_ssim_loss = 1 - pytorch_ssim.ssim(spa_cover / 255.0, spa_adv_stego / 255.0)
            g_vgg_loss = torch.tensor(0.)
            if args.use_vgg:
                vgg_on_cov = vgg(spa_cover / 255.0)
                vgg_on_enc = vgg(spa_adv_stego / 255.0)
                g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)
            g_decoder_loss = bce_loss(extract_message, message_in)

            g_loss = (g_adv_loss + g_mse_loss + 100 * g_decoder_loss + g_ssim_loss + g_vgg_loss) \
                if use_discriminator else (g_mse_loss + 100 * g_decoder_loss + g_vgg_loss)
            g_loss.backward()
            optimizer_coders.step()
            with torch.no_grad():
                decoded_rounded = extract_message.detach().cpu().numpy().round().clip(0, 1)
                decode_accuracy = (decoded_rounded == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
                bit_err = 1 - decode_accuracy
                image_in_round = torch.round(spa_cover)
                stego_round = torch.round(spa_adv_stego)

                metrics['a_loss'] += a_loss.item()
                metrics['adv_loss'] += g_adv_loss.item()
                metrics['mse_loss'] += g_mse_loss.item()
                metrics['ssim_loss'] += g_ssim_loss.item()
                metrics['vgg_loss'] += g_vgg_loss.item()
                metrics['decoder_loss'] += g_decoder_loss.item()
                metrics['loss'] += g_loss.item()
                metrics['decode_accuracy'] += decode_accuracy.item()
                metrics['bit_err'] += bit_err.item()
                metrics['psnr'] += utils.psnr_between_batches(image_in_round, stego_round)
                metrics['ssim'] += pytorch_ssim.ssim(image_in_round / 255.0, stego_round / 255.0).item()

            i += 1
            iteration += 1


            if iteration % 50 == 0:
                for k in metrics.keys():
                    metrics[k] /= 50
                print('\nEpoch: %d, iteration: %d' % (e, iteration))
                for k in metrics.keys():
                    if 'loss' in k:
                        print(k + ': %.6f' % metrics[k], end='\t')
                print('')
                for k in metrics.keys():
                    if 'loss' not in k:
                        print(k + ': %.6f' % metrics[k], end='\t')
                utils.write_losses(train_csv_file, iteration, e, metrics, time.time() - tic)
                for k in metrics.keys():
                    metrics[k] = 0

                adv_0_img = torch.where(adv_0 < 0.95, torch.zeros_like(adv_0), torch.ones_like(adv_0)).clone().detach()
                print('l0:', torch.norm(adv_0_img, 0) / args.batch_size)


        # ------ validate ------
        val_metrics = {m: 0 for m in metric_names}
        print('\n---> Epoch %d starts validating...' % e)
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        netG.eval()
        correct1 = 0
        correct2 = 0
        for batch_id, image_in in enumerate(valid_loader):
            image_in = image_in.cuda()
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            # message_in = next(iter(message_loader))

            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()

            with torch.no_grad():
                stego = encoder(image_in, message_in)
                adv_stego, adv_inf, adv_0, adv_00 = netG(stego)
                extract_message = decoder(adv_stego)
                adv_stego1 = torch.round(adv_stego).permute(0, 2, 3, 1)
                spa_adv_stego = get_spatial(adv_stego1).cuda()
                spa_adv_stego_round = torch.round(spa_adv_stego).clamp(0, 255)
                ycbcr_adv_stego = get_ycbcr(adv_stego1).permute(0, 3, 1, 2).cuda()
                output1 = netT1(ycbcr_adv_stego)
                output2 = netT2(spa_adv_stego_round)

                curr_adv_label1 = output1.max(1, keepdim=True)[1]  # 找出最大的下标
                targ_adv_label1 = torch.zeros_like(curr_adv_label1).cuda()  # 0是cover，1是stego
                curr_adv_pred1 = output1.gather(1, curr_adv_label1)
                targ_adv_pred1 = output1.gather(1, targ_adv_label1)
                attack_loss1 = torch.mean(curr_adv_pred1 - targ_adv_pred1)

                curr_adv_label2 = output2.max(1, keepdim=True)[1]  # 找出最大的下标
                targ_adv_label2 = torch.zeros_like(curr_adv_label2).cuda()  # 0是cover，1是stego
                curr_adv_pred2 = output2.gather(1, curr_adv_label2)
                targ_adv_pred2 = output2.gather(1, targ_adv_label2)
                attack_loss2 = torch.mean(curr_adv_pred2 - targ_adv_pred2)

                attack_loss = (attack_loss1 + attack_loss2) / 2

                spa_loss = torch.norm(adv_0, 1)
                bi_adv_00 = torch.where(adv_00 < 0.95, torch.zeros_like(adv_00), torch.ones_like(adv_00))
                qua_loss = torch.sum((bi_adv_00 - adv_00) ** 2)
                a_loss = attack_loss + 0.0001 * spa_loss + 0.0001 * qua_loss

                image_in1 = image_in.permute(0, 2, 3, 1)
                stego1 = stego.permute(0, 2, 3, 1)
                spa_stego = get_spatial(stego1).cuda()  # spatial domain
                spa_cover = get_spatial(image_in1).cuda()
                spa_cover_round = torch.round(spa_cover).clamp(0, 255)

                M = (M_texture(spa_cover) + M_color(spa_cover)) / 2

                diff = norm(spa_cover) - norm(spa_adv_stego)

                eye_loss = torch.mean(torch.abs(M * diff))

                g_on_adv_stego = discriminator(spa_adv_stego)

                g_adv_loss = - g_on_adv_stego.mean()
                g_mse_loss = mse_loss(spa_adv_stego, spa_cover) + eye_loss
                g_ssim_loss = 1 - pytorch_ssim.ssim(spa_cover / 255.0, spa_adv_stego / 255.0)
                g_vgg_loss = torch.tensor(0.)
                if args.use_vgg:
                    vgg_on_cov = vgg(spa_cover / 255.0)
                    vgg_on_enc = vgg(spa_adv_stego / 255.0)
                    g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)

                g_decoder_loss = bce_loss(extract_message, message_in)
                g_loss = (g_adv_loss + g_mse_loss + 100 * g_decoder_loss + g_ssim_loss + g_vgg_loss) \
                    if use_discriminator else (g_mse_loss + 100 * g_decoder_loss + g_vgg_loss)

                decoded_rounded = extract_message.detach().cpu().numpy().round().clip(0, 1)
                decode_accuracy = (decoded_rounded == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
                bit_err = 1 - decode_accuracy

                pred1 = output1.max(1, keepdim=True)[1]  # 找到概率最大的下标
                pred2 = output2.max(1, keepdim=True)[1]
                target1 = torch.ones_like(pred1).cuda()
                target2 = torch.ones_like(pred2).cuda()
                steganalyzer_acc1 = pred1.eq(target1.view_as(pred1)).sum().item()
                steganalyzer_acc2 = pred2.eq(target2.view_as(pred2)).sum().item()
                correct1 += steganalyzer_acc1
                correct2 += steganalyzer_acc2

                val_metrics['a_loss'] += a_loss.item()
                val_metrics['adv_loss'] += g_adv_loss.item()
                val_metrics['mse_loss'] += g_mse_loss.item()
                val_metrics['ssim_loss'] += g_ssim_loss.item()
                val_metrics['vgg_loss'] += g_vgg_loss.item()
                val_metrics['decoder_loss'] += g_decoder_loss.item()
                val_metrics['loss'] += g_loss.item()
                val_metrics['decode_accuracy'] += decode_accuracy.item()
                val_metrics['bit_err'] += bit_err.item()
                val_metrics['psnr'] += utils.psnr_between_batches(spa_cover_round, spa_adv_stego_round)
                val_metrics['ssim'] += pytorch_ssim.ssim(spa_cover_round / 255.0, spa_adv_stego_round / 255.0).item()

        print('\nPE1: {:.4f}\n'.format(1 - correct1 / len(valid_loader.dataset)))
        print('\nPE2: {:.4f}\n'.format(1 - correct2 / len(valid_loader.dataset)))

        adv_0_img = torch.where(adv_0 < 0.95, torch.zeros_like(adv_0), torch.ones_like(adv_0)).clone().detach()
        print('l0:', torch.norm(adv_0_img, 0) / args.batch_size)

        for k in val_metrics.keys():
            val_metrics[k] /= len(valid_loader)
        print('Valid epoch: {}'.format(e))
        for k in val_metrics.keys():
            if 'loss' in k:
                print(k + ': %.6f' % val_metrics[k], end='\t')
        print('')
        for k in val_metrics.keys():
            if 'loss' not in k:
                print(k + ': %.6f' % val_metrics[k], end='\t')
        print('time:%.0f' % (time.time() - tic))
        print('Epoch %d finished, taking %0.f seconds\n' % (e, time.time() - epoch_start_time))
        utils.write_losses(valid_csv_file, iteration, e, val_metrics, time.time() - tic)

        scheduler.step()

        # save model
        if (e + 1) % 1 == 0 or e == args.epochs - 1:
            checkpoint = {
                'epoch': e,
                'iteration': iteration,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': netG.state_dict(),
                'optimizer_coders_state_dict': optimizer_coders.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                'optimizer_generator_state_dict': optimizer_netG.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics
            }
            filename = os.path.join(checkpoints_dir, "epoch_" + str(e) + ".pt")
            torch.save(checkpoint, filename)


if __name__ == '__main__':
    main()
