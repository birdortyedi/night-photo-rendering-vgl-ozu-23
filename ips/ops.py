import math
import numpy as np
import cv2

from fractions import Fraction
from exifread.utils import Ratio
from skimage import img_as_ubyte
from scipy import signal
from torch.nn import functional as F

from ips.wb import *
from ips.gamma import *
from ips.autocontrast import *
from utils import misc, color


def linearize_raw(raw_img, img_meta):
    return raw_img


def normalize_(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = misc.ratios2floats(black_level)
        if type(black_level[0]) is Fraction:
            black_level = misc.fractions2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    normalized_image[normalized_image > 1] = 1
    return normalized_image


def normalize(linearized_raw, img_meta):
    return normalize_(linearized_raw, img_meta['black_level'], img_meta['white_level'])


def bad_pixel_correction(data, neighborhood_size=3):
    if (neighborhood_size % 2) == 0:
        print("neighborhood_size shoud be odd number, recommended value 3")
        return data

    # convert to float32 in case they were not
    # Being consistent in data format to be float32
    data = np.float32(data)

    # Separate out the quarter resolution images
    D = {0: data[::2, ::2], 1: data[::2, 1::2], 2: data[1::2, ::2], 3: data[1::2, 1::2]}  # Empty dictionary

    # number of pixels to be padded at the borders
    no_of_pixel_pad = math.floor(neighborhood_size / 2.)

    for idx in range(0, len(D)):  # perform same operation for each quarter
        img = D[idx]
        height, width = img.shape

        # pad pixels at the borders
        img = np.pad(img, (no_of_pixel_pad, no_of_pixel_pad), 'reflect')  # reflect would not repeat the border value

        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # save the middle pixel value
                mid_pixel_val = img[i, j]

                # extract the neighborhood
                neighborhood = img[i - no_of_pixel_pad: i + no_of_pixel_pad + 1, j - no_of_pixel_pad: j + no_of_pixel_pad + 1]

                # set the center pixels value same as the left pixel
                # Does not matter replace with right or left pixel
                # is used to replace the center pixels value
                neighborhood[no_of_pixel_pad, no_of_pixel_pad] = neighborhood[no_of_pixel_pad, no_of_pixel_pad - 1]

                min_neighborhood = np.min(neighborhood)
                max_neighborhood = np.max(neighborhood)

                if mid_pixel_val < min_neighborhood:
                    img[i, j] = min_neighborhood
                elif mid_pixel_val > max_neighborhood:
                    img[i, j] = max_neighborhood
                else:
                    img[i, j] = mid_pixel_val

        # Put the corrected image to the dictionary
        D[idx] = img[no_of_pixel_pad: height + no_of_pixel_pad, no_of_pixel_pad: width + no_of_pixel_pad]

    # Regrouping the data
    data[::2, ::2] = D[0]
    data[::2, 1::2] = D[1]
    data[1::2, ::2] = D[2]
    data[1::2, 1::2] = D[3]

    return data
  

def white_balance(input_img, illumuniation_estimation_algorithm):
    as_shot_neutral = illumination_parameters_estimation(input_img, illumuniation_estimation_algorithm)    
    
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)

    as_shot_neutral = np.asarray(as_shot_neutral)
    # transform vector into matrix
    if as_shot_neutral.shape == (3,):
        as_shot_neutral = np.diag(1./as_shot_neutral)

    assert as_shot_neutral.shape == (3, 3)

    white_balanced_image = np.dot(input_img, as_shot_neutral.T)
    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)

    return white_balanced_image


def denoise_image(demosaiced_image, method="bilateral"):
    assert method in ["bilateral", "wavelet", "tv"]
    if method == "bilateral":
        from skimage.restoration import denoise_bilateral
        current_image = denoise_bilateral(
            demosaiced_image, sigma_color=None, sigma_spatial=1., channel_axis=-1, mode='reflect')
    elif method == "wavelet":
        from skimage.restoration import denoise_wavelet
        current_image = denoise_wavelet(
            demosaiced_image, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
    else:
        raise NotImplementedError()
    return current_image


def apply_gamma(x, method="orj"):
    if method == "orj":
        return apply_gamma_orj(x)
    elif method == "channel":
        return apply_gamma_channel_wise(x)
    else:
        return apply_gamma_base(x)


def perform_autocontrast(img, method):
    assert method in ["pil", "standard", "channel1", "channel2"]
    if method == "pil":
        return autocontrast_using_pil(img)
    elif method == "standard":
        return perform_autocontrast_standard(img, cutoff=(4, 0))
    elif method == "old":
        return perform_autocontrast_channel1(img)
    else:
        return perform_autocontrast_channel2(img)
    

def white_balance_corr(demosaic, net, wb_settings, multi_scale=True, post_process=True, device="cpu"):
    return white_balance_corr_style(demosaic, net, wb_settings, multi_scale=multi_scale, post_process=post_process, device=device)


def memory_color_enhancement(data, color_space="srgb", illuminant="d65", clip_range=[0, 255], cie_version="1931"):
    target_hue = [30., -125., 100.]
    hue_preference = [20., -118., 130.]
    hue_sigma = [20., 10., 5.]
    is_both_side = [True, False, False]
    multiplier = [0.6, 0.6, 0.6]
    chroma_preference = [25., 14., 30.]
    chroma_sigma = [10., 10., 5.]

    # RGB to xyz
    data = color.rgb2xyz(data, color_space, clip_range)
    # xyz to lab
    data = color.xyz2lab(data, cie_version, illuminant)
    # lab to lch
    data = color.lab2lch(data)

    # hue squeezing
    # we are traversing through different color preferences
    height, width, _ = data.shape
    hue_correction = np.zeros((height, width), dtype=np.float32)
    for i in range(0, np.size(target_hue)):

        delta_hue = data[:, :, 2] - hue_preference[i]

        if is_both_side[i]:
            weight_temp = np.exp(-np.power(data[:, :, 2] - target_hue[i], 2) / (2 * hue_sigma[i] ** 2)) + \
                          np.exp(-np.power(data[:, :, 2] + target_hue[i], 2) / (2 * hue_sigma[i] ** 2))
        else:
            weight_temp = np.exp(-np.power(data[:, :, 2] - target_hue[i], 2) / (2 * hue_sigma[i] ** 2))

        weight_hue = multiplier[i] * weight_temp / np.max(weight_temp)

        weight_chroma = np.exp(-np.power(data[:, :, 1] - chroma_preference[i], 2) / (2 * chroma_sigma[i] ** 2))

        hue_correction = hue_correction + np.multiply(np.multiply(delta_hue, weight_hue), weight_chroma)

    # correct the hue
    data[:, :, 2] = data[:, :, 2] - hue_correction

    # lch to lab
    data = color.lch2lab(data)
    # lab to xyz
    data = color.lab2xyz(data, cie_version, illuminant)
    # xyz to rgb
    data = color.xyz2rgb(data, color_space, clip_range)

    data = misc.outOfGamutClipping(data, range=clip_range[1])
    return data


def to_uint8(srgb):
    return (srgb * 255).astype(np.uint8)


def resize_using_pil(img, width=1296, height=864):
    img_pil = Image.fromarray(img)
    out_size = (width, height)
    if img_pil.size == out_size:
        return img
    out_img = img_pil.resize(out_size, Image.ANTIALIAS)
    out_img = np.array(out_img)
    return out_img


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def unsharp_masking(data, gaussian_kernel_size=[5, 5], gaussian_sigma=2.0, slope=1.5, tau_threshold=0.05, gamma_speed=4., clip_range=[0, 255]):
    # create gaussian kernel
    gaussian_kernel = misc.gaussian(gaussian_kernel_size, gaussian_sigma)

    # convolve the image with the gaussian kernel
    # first input is the image
    # second input is the kernel
    # output shape will be the same as the first input
    # boundary will be padded by using symmetrical method while convolving
    if np.ndim(data) > 2:
        image_blur = np.empty(np.shape(data), dtype=np.float32)
        for i in range(0, np.shape(data)[2]):
            image_blur[:, :, i] = signal.convolve2d(data[:, :, i], gaussian_kernel, mode="same", boundary="symm")
    else:
        image_blur = signal.convolve2d(data, gaussian_kernel, mode="same", boundary="symm")

    # the high frequency component image
    image_high_pass = data - image_blur

    # soft coring (see in utility)
    # basically pass the high pass image via a slightly nonlinear function
    tau_threshold = tau_threshold * clip_range[1]

    # add the soft cored high pass image to the original and clip
    # within range and return
    def soft_coring(img_hp, slope, tau_threshold, gamma_speed):
        return slope * np.float32(img_hp) * (1. - np.exp(-((np.abs(img_hp / tau_threshold))**gamma_speed)))
    return np.clip(data + soft_coring(image_high_pass, slope, tau_threshold, gamma_speed), clip_range[0], clip_range[1])


def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def denoise_postprocess(img, net, device, img_multiple_of=8, grid_size=5):
    input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device=device)
    
    _, _, h, w = input_.shape
    hp = h // grid_size
    wp = w // grid_size
    
    restored = torch.zeros_like(input_)
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                patch = input_[:, :, i*hp:(i+1)*hp, j*wp:(j+1)*wp]            
                # Pad the input if not_multiple_of 8
                h,w = patch.shape[2], patch.shape[3]
                H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
                padh = H-h if h%img_multiple_of!=0 else 0
                padw = W-w if w%img_multiple_of!=0 else 0
                patch = F.pad(patch, (0,padw,0,padh), 'reflect')
                
                restored[:, :, i*hp:(i+1)*hp, j*wp:(j+1)*wp] = net(patch)[:,:,:hp,:wp] # Unpad the output
        
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
    return restored
