# -*- coding:utf-8 -*-
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time
import random
from skimage import io
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
import cv2
import scipy.io as sio
from jpeg.compression import rgb_to_ycbcr_jpeg
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Testing of our nets')
    parser.add_argument('--test-data-dir', required=True, type=str,
                        help='The directory where the data for testing is stored.')
    parser.add_argument('--checkpoint-path', required=True, type=str,
                        help='The path to the checkpoint.')
    parser.add_argument('--test-stego-dir', type=str, default='',
                        help='The path where stego images will be stored.')
    parser.add_argument('--no-save-stego', action='store_true', default=False,
                        help='Do not save stego images.')

    parser.add_argument('--batch-size', type=int, help='The batch size.', default=4)
    parser.add_argument('--data-depth', default=1, type=int, help='The depth of the message.')
    parser.add_argument('--gray', action='store_true', default=False,
                        help='Use gray-scale images.')
    parser.add_argument('--hidden-size', type=int, default=32,
                        help='Hidden channels in networks.')
    parser.add_argument('--seed', type=int, default=20,
                        help='Random seed.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Index of gpu used (default: 0).')
    parser.add_argument('--use-vgg', action='store_true', default=False,
                        help='Use VGG loss.')
    parser.add_argument('--size', default=256, type=int,
                        help='The size of the images (images are square so this is height and width).')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
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
    test_transform = transforms.Compose([
        # transforms.ToTensor()
    ])
    test_dataset = utils.Mydataset(args.test_data_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)

    # test_message_path = '/data2/as/nyx/data/qr_dataset_real256'
    # test_message = utils.Mymessage(test_message_path)
    # test_message_loader = DataLoader(test_message, batch_size=args.batch_size, drop_last=False, shuffle=True, **kwargs)


    # Load Models
    print('---> Constructing Network Architecture'
          'res...')
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

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    netG.load_state_dict(checkpoint['generator_state_dict'])

    steganalyzer_para1 = '/data2/as/nyx/UCNet_Steganalysis-main/UCNet_JPEG/0.4-0.02-params-lr=75.pt'
    all_state1 = torch.load(steganalyzer_para1)
    original_state1 = all_state1['original_state']
    netT1.load_state_dict(original_state1)
    netT1.eval()

    steganalyzer_para2 = '/data2/as/nyx/UCNet_Steganalysis-main/UCNet_Spatial/params-lr=0.01.pt'
    all_state2 = torch.load(steganalyzer_para2)
    original_state2 = all_state2['original_state']
    netT2.load_state_dict(original_state2)
    netT2.eval()

    for p in netT1.parameters():
        p.requires_grad = False

    for p in netT2.parameters():
        p.requires_grad = False


    # loss
    print('---> Constructing VGG-16 for Perceptual Loss...')
    vgg = utils.VGGLoss(3, 1, False)
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

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

    metric_names = ['a_loss', 'adv_loss', 'mse_loss', 'ssim_loss', 'vgg_loss', 'decoder_loss', 'loss',
                    'bit_err', 'decode_accuracy', 'psnr', 'ssim']
    metrics = {m: 0 for m in metric_names}
    idx = 0
    file_names = os.listdir(args.test_data_dir)
    name = []
    for names in file_names:
        name.append(names[0:-4])
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    netG.eval()
    for batch_id, image_in in enumerate(test_loader):
        with torch.no_grad():
            image_in = image_in.cuda()
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            # message_in = next(iter(test_message_loader))

            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()

            stego = encoder(image_in, message_in)
            adv_stego, adv_inf, adv_0, adv_00 = netG(stego)
            extract_message = decoder(adv_stego)
            image_in1 = image_in.permute(0, 2, 3, 1)
            stego1 = stego.permute(0, 2, 3, 1)
            spa_stego = get_spatial(stego1).cuda()  # spatial domain
            spa_cover = get_spatial(image_in1).cuda()
            adv_stego1 = torch.round(adv_stego).permute(0, 2, 3, 1)
            spa_adv_stego = get_spatial(adv_stego1).cuda()
            spa_cover_round = torch.round(spa_cover)
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
            d_loss = mse_loss(adv_stego, stego)
            a_loss = attack_loss + 0.0001 * spa_loss + 0.0001 * qua_loss + 0.01 * d_loss

            d_on_stego = discriminator(spa_adv_stego)
            M = (M_texture(spa_cover) + M_color(spa_cover)) / 2
            diff = norm(spa_cover) - norm(spa_adv_stego)

            eye_loss = torch.mean(torch.abs(M * diff))

            g_adv_loss = d_on_stego.mean()
            g_mse_loss = mse_loss(spa_adv_stego, spa_cover) + eye_loss
            g_vgg_loss = torch.tensor(0.)
            if args.use_vgg:
                vgg_on_cov = vgg(spa_cover / 255.0)
                vgg_on_enc = vgg(spa_adv_stego / 255.0)
                g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)

            g_decoder_loss = bce_loss(extract_message, message_in)
            g_loss = (g_adv_loss + g_mse_loss + 100 * g_decoder_loss + g_vgg_loss)

            decoded_rounded = extract_message.detach().cpu().numpy().round().clip(0, 1)
            decode_accuracy = (decoded_rounded == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
            bit_err = 1 - decode_accuracy

        metrics['a_loss'] += a_loss.item()
        metrics['adv_loss'] += g_adv_loss.item()
        metrics['mse_loss'] += g_mse_loss.item()
        metrics['vgg_loss'] += g_vgg_loss.item()
        metrics['decoder_loss'] += g_decoder_loss.item()
        metrics['loss'] += g_loss.item()
        metrics['decode_accuracy'] += decode_accuracy.item()
        metrics['bit_err'] += bit_err.item()
        metrics['psnr'] += utils.psnr_between_batches(spa_cover_round, spa_adv_stego_round)
        metrics['ssim'] += pytorch_ssim.ssim(spa_cover_round / 255.0, spa_adv_stego_round / 255.0).item()

        # if one wants to save stego images
        if not args.no_save_stego:
            stego_path = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint_path)), 'stegos') if \
                args.test_stego_dir == '' else args.test_stego_dir
            if not os.path.exists(stego_path):
                os.makedirs(stego_path)
            for idx_in_batch in range(batch_size):
                name[idx] = name[idx] + '.png'
                file_path = os.path.join(stego_path, name[idx])
                image = spa_adv_stego_round[idx_in_batch].permute(1, 2, 0)
                imagenp = image.detach().cpu().numpy().astype('uint8')
                # imagenp = image.detach().cpu().numpy()
                io.imsave(file_path, imagenp)
                # sio.savemat(file_path, {'img':imagenp})
                idx += 1

    for k in metrics.keys():
        metrics[k] /= len(test_loader)

    for k in metrics.keys():
        print(k + ': %.6f' % metrics[k])

if __name__ == '__main__':
    main()
