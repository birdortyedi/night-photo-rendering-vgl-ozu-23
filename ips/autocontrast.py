import numpy as np
from PIL import Image, ImageOps


def autocontrast_using_pil(img, cutoff=3):
    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil = ImageOps.autocontrast(img_pil, cutoff=cutoff)
    output_image = np.array(img_pil).astype(np.float32) / 255.
    return output_image


def _lut(image, lut):
    if image.mode == "P":
        # FIXME: apply to lookup table, not image data
        raise NotImplementedError("mode P support coming soon")
    elif image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        raise OSError("not supported for this image mode")
    
    
def autocontrast(image, cutoff=(0, 0), ignore=None):
    """
    Maximize (normalize) image contrast. This function calculates a
    histogram of the input image, removes **cutoff** percent of the
    lightest and darkest pixels from the histogram, and remaps the image
    so that the darkest pixel becomes black (0), and the lightest
    becomes white (255).

    :param image: The image to process.
    :param cutoff: How many percent to cut off from the histogram.
    :param ignore: The background pixel value (use None for no background).
    :return: An image.
    """
    histogram = image.histogram()
    lut = []
    for layer in range(0, len(histogram), 256):
        h = histogram[layer : layer + 256]
        if ignore is not None:
            # get rid of outliers
            try:
                h[ignore] = 0
            except TypeError:
                # assume sequence
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the hi end
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut.extend(list(range(256)))
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                ix = int(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut.append(ix)
    return _lut(image, lut)


def perform_autocontrast_standard(img, cutoff=(2, 0)):
    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil = autocontrast(img_pil, cutoff=cutoff)
    output_image = np.array(img_pil).astype(np.float32) / 255.
    return output_image


def perform_autocontrast_channel1(img):
    
    def reject_outliers(data, m=1.2):
        return abs(data - np.mean(data)) < m * np.std(data)
    
    def get_cutoff(img_ch):
        values, _ = np.histogram(img_ch, bins=32)
        ratios = values / values.sum()
        cutoff = 4 if reject_outliers(values)[0] else 4 - np.log(100 * np.abs(ratios[1]-ratios[0]))
        if cutoff < 0:
            cutoff = 0
        return int(cutoff)

    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    r, g, b = img_pil.split()
    cutoff_r = get_cutoff(np.array(r).flatten())
    cutoff_g = get_cutoff(np.array(g).flatten())
    cutoff_b = get_cutoff(np.array(b).flatten())
    r_ = autocontrast(r, cutoff=(cutoff_r, 0))
    g_ = autocontrast(g, cutoff=(cutoff_g, 0))
    b_ = autocontrast(b, cutoff=(cutoff_b, 0))
    output_r = np.array(r_).astype(np.float32) / 255.
    output_g = np.array(g_).astype(np.float32) / 255.
    output_b = np.array(b_).astype(np.float32) / 255.
    output_image = np.transpose(np.array([output_r, output_g, output_b]), (1, 2, 0))
    return output_image


def perform_autocontrast_channel2(img):
    
    def get_cutoff(img_uint8, base_cutoff=4):
        cutoff = list()
        h, w, _ = img_uint8.shape
        for ch in Image.fromarray(img_uint8).split():
            values, _ = np.histogram(np.array(ch).flatten(), bins=32)
            cutoff.append(np.ceil((values.cumsum() / (h * w))[0] * 100).astype(int))
        cutoff = [coff if coff > base_cutoff else base_cutoff for coff in cutoff]
        return cutoff

    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    cutoff = get_cutoff(img_uint8)
    output = np.array([
        np.array(autocontrast(ch, cutoff=(coff, 0))).astype(np.float32) / 255. 
        for ch, coff in zip(Image.fromarray(img_uint8).split(), cutoff)
    ])
    return np.transpose(output, (1, 2, 0))
