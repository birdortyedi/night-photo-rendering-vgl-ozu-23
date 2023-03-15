from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from colour import cctf_encoding

from utils import misc
from ips.ops import *


def process(raw_image, metadata, wb_network, restormer):
    img = linearize_raw(raw_image, metadata)  # linearization tables for all inputs are null, so just return the input.
    out = normalize(img, metadata)  # black level correction according to the metadata
    out = bad_pixel_correction(out)
    out = demosaicing_CFA_Bayer_Menon2007(out, misc.decode_cfa_pattern(metadata['cfa_pattern']))
    wb_img = white_balance(out, "iwp")
    out = cctf_encoding(wb_img)
    srgb = misc.outOfGamutClipping(out)
    out = denoise_image(srgb, metadata["denoise_method"])
    gamma = apply_gamma(out, metadata["correction_method"])
    autoc = perform_autocontrast(gamma, metadata["autocontrast_method"])
    out = white_balance_corr(autoc, wb_network, metadata["wb_settings"], True, True, device=wb_network.device)
    out = memory_color_enhancement(out, clip_range=[0, 1])
    out = to_uint8(out)
    out = resize_using_pil(out, metadata["exp_width"], metadata["exp_height"])
    out = fix_orientation(out, metadata['orientation'])
    if restormer is not None:
        out = denoise_postprocess(out, restormer, device="cuda:1")
    out = unsharp_masking(out / 255.)
    out = adjust_contrast_brightness(out * 255., contrast=1.2, brightness=16)
    return out
