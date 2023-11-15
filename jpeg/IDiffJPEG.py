# Pytorch
import torch
import torch.nn as nn
# Local
from jpeg.decompression import decompress_jpeg
from jpeg.utils import diff_round, quality_to_factor
from jpeg.ycbcr import get_ycbcr

class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        
        return image[:, :, :, 0], image[:, :, :, 1], image[:, :, :, 2]

class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output:
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)

class IDCT(nn.Module):
    def __init__(self, height, width, differentiable=False, quality=75):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(IDCT, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = diff_round
        factor = quality_to_factor(quality)
        self.decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor)
        self.chroma = chroma_subsampling()
        self.splitting = block_splitting()

        self.height, self.width = height, width

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.chroma(x)
        y = self.splitting(y)
        cb = self.splitting(cb)
        cr = self.splitting(cr)
        #
        image = self.decompress(y, cb, cr)

        return image


class YCbCr_Space(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=75):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(YCbCr_Space, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = diff_round
        factor = quality_to_factor(quality)
        self.decompress = get_ycbcr(height, width, rounding=rounding, factor=factor)
        self.chroma = chroma_subsampling()
        self.splitting = block_splitting()

        self.height, self.width = height, width

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.chroma(x)
        y = self.splitting(y)
        cb = self.splitting(cb)
        cr = self.splitting(cr)
        #
        image = self.decompress(y, cb, cr)

        return image


# if __name__ == '__main__':
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     test_sample = cv2.imread('/home/SAE-JS/00001.tif')
#     test_sample = cv2.cvtColor(test_sample, cv2.COLOR_BGR2RGB)
#     print(test_sample.shape)
#     print(test_sample)
#     jpeg_nn = compress_coefficients(height=256, width=256, differentiable=True, quality=75).to(device)
#     trans = transforms.ToTensor()
#     test_sample = trans(test_sample).to(device)
#     test_sample = torch.unsqueeze(test_sample, 0)
#     # print(test_sample.size())
#     # print(test_sample)
#     out_sample = jpeg_nn(test_sample).to(device)
#     print(out_sample.shape)
#     toPIL = transforms.ToPILImage()
#     out_sample = torch.squeeze(out_sample, 0)
#     out_sample = toPIL(out_sample)
#     out_sample.save('../0_jpeg.jpg')
