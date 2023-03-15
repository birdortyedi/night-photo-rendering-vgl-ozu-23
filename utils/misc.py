import numpy as np
from math import ceil
import torch
from modeling.DeepWB.utilities import imresize


def decode_cfa_pattern(cfa_pattern):
    cfa_dict = {0: 'B', 1: 'G', 2: 'R'}
    return "".join([cfa_dict[x] for x in cfa_pattern])


def to_tensor(im, dims=3):
    """ Converts a given ndarray image to torch tensor image.

  Args:
    im: ndarray image (height x width x channel x [sample]).
    dims: dimension number of the given image. If dims = 3, the image should
      be in (height x width x channel) format; while if dims = 4, the image
      should be in (height x width x channel x sample) format; default is 3.

  Returns:
    torch tensor in the format (channel x height x width)  or (sample x
      channel x height x width).
  """

    assert (dims == 3 or dims == 4)
    if dims == 3:
        im = im.transpose((2, 0, 1))
    elif dims == 4:
        im = im.transpose((0, 3, 1, 2))
    else:
        raise NotImplementedError

    return torch.from_numpy(im.copy())


def outOfGamutClipping(I, range=1.):
    """ Clips out-of-gamut pixels. """
    if range == 1.:
        I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
        I[I < 0] = 0  # any pixel is below 0, clip it to 0
    else:
        I[I > 255] = 255  # any pixel is higher than 255, clip it to 255
        I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def fractions2floats(fractions):
    floats = []
    for fraction in fractions:
        floats.append(float(fraction.numerator) / fraction.denominator)
    return floats


def gaussian(kernel_size, sigma):
    # calculate which number to where the grid should be
    # remember that, kernel_size[0] is the width of the kernel
    # and kernel_size[1] is the height of the kernel
    temp = np.floor(np.float32(kernel_size) / 2.)

    # create the grid
    # example: if kernel_size = [5, 3], then:
    # x: array([[-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.]])
    # y: array([[-1., -1., -1., -1., -1.],
    #           [ 0.,  0.,  0.,  0.,  0.],
    #           [ 1.,  1.,  1.,  1.,  1.]])
    x, y = np.meshgrid(np.linspace(-temp[0], temp[0], kernel_size[0]), np.linspace(-temp[1], temp[1], kernel_size[1]))

    # Gaussian equation
    temp = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    # make kernel sum equal to 1
    return temp / np.sum(temp)


def aspect_ratio_imresize(im, max_output=256):
    h, w, c = im.shape
    if max(h, w) > max_output:
        ratio = max_output / max(h, w)
        im = imresize.imresize(im, scalar_scale=ratio)
        h, w, c = im.shape

    if w % (2 ** 4) == 0:
        new_size_w = w
    else:
        new_size_w = w + (2 ** 4) - w % (2 ** 4)

    if h % (2 ** 4) == 0:
        new_size_h = h
    else:
        new_size_h = h + (2 ** 4) - h % (2 ** 4)

    new_size = (new_size_h, new_size_w)
    if not ((h, w) == new_size):
        im = imresize.imresize(im, output_shape=new_size)

    return im


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg
    

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg
    

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B