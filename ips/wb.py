import numpy as np
import torch

from kornia.geometry.transform import resize

from modeling.DeepWB.arch import deep_wb_single_task as dwb
from modeling.DeepWB.utilities.deepWB import deep_wb
from modeling.DeepWB.utilities.utils import colorTempInterpolate_w_target
from modeling import weight_refinement
from utils import misc, optim


def illumination_parameters_estimation(current_image, illumination_estimation_option):
    ie_method = illumination_estimation_option.lower()
    if ie_method == "gw":
        ie = np.mean(current_image, axis=(0, 1))
        ie /= ie[1]
        return ie
    elif ie_method == "sog":
        sog_p = 4.
        ie = np.mean(current_image**sog_p, axis=(0, 1))**(1/sog_p)
        ie /= ie[1]
        return ie
    elif ie_method == "wp":
        ie = np.max(current_image, axis=(0, 1))
        ie /= ie[1]
        return ie
    elif ie_method == "iwp":
        samples_count = 20
        sample_size = 20
        rows, cols = current_image.shape[:2]
        data = np.reshape(current_image, (rows*cols, 3))
        maxima = np.zeros((samples_count, 3))
        for i in range(samples_count):
            maxima[i, :] = np.max(data[np.random.randint(
                low=0, high=rows*cols, size=(sample_size)), :], axis=0)
        ie = np.mean(maxima, axis=0)
        ie /= ie[1]
        return ie
    else:
        raise ValueError(
            'Bad illumination_estimation_option value! Use the following options: "gw", "wp", "sog", "iwp"')
      
      
def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def white_balance_corr_style(demosaic, net, wb_settings, multi_scale=True, post_process=True, device="cpu"):
    size = demosaic.shape[:2]

    deepWB_T = dwb.deepWBnet()
    deepWB_T.load_state_dict(torch.load('modeling/DeepWB/models/net_t.pth'))
    deepWB_S = dwb.deepWBnet()
    deepWB_S.load_state_dict(torch.load('modeling/DeepWB/models/net_s.pth'))
    deepWB_T.eval().to(device)
    deepWB_S.eval().to(device)

    t_img, s_img = deep_wb(demosaic, task='editing', net_s=deepWB_S, net_t=deepWB_T, device=device)
    full_size_img = demosaic.copy()
    d_img = misc.imresize(demosaic, output_shape=(512, 512))
    t_img = misc.imresize(t_img, output_shape=(512, 512))
    s_img = misc.imresize(s_img, output_shape=(512, 512))
    s_mapping = optim.get_mapping_func(d_img, s_img)
    t_mapping = optim.get_mapping_func(d_img, t_img)
    full_size_s = optim.apply_mapping_func(full_size_img, s_mapping)
    full_size_s = misc.outOfGamutClipping(full_size_s)
    full_size_t = optim.apply_mapping_func(full_size_img, t_mapping)
    full_size_t = misc.outOfGamutClipping(full_size_t)
    if 'F' in wb_settings:
        f_img = colorTempInterpolate_w_target(t_img, s_img, 3800)
        f_mapping = optim.get_mapping_func(d_img, f_img)
        full_size_f = optim.apply_mapping_func(full_size_img, f_mapping)
        full_size_f = misc.outOfGamutClipping(full_size_f)
    else:
        f_img = None

    if 'C' in wb_settings:
        c_img = colorTempInterpolate_w_target(t_img, s_img, 6500)
        c_mapping = optim.get_mapping_func(d_img, c_img)
        full_size_c = optim.apply_mapping_func(full_size_img, c_mapping)
        full_size_c = misc.outOfGamutClipping(full_size_c)
    else:
        c_img = None
    d_img = misc.to_tensor(d_img, dims=3)
    s_img = misc.to_tensor(s_img, dims=3)
    t_img = misc.to_tensor(t_img, dims=3)
    if f_img is not None:
        f_img = misc.to_tensor(f_img, dims=3)
    if c_img is not None:
        c_img = misc.to_tensor(c_img, dims=3)
    img = torch.cat((d_img, s_img, t_img), dim=0)
    if f_img is not None:
        img = torch.cat((img, f_img), dim=0)
    if c_img is not None:
        img = torch.cat((img, c_img), dim=0)
    full_size_img = misc.to_tensor(full_size_img, dims=3)
    full_size_s = misc.to_tensor(full_size_s, dims=3)
    full_size_t = misc.to_tensor(full_size_t, dims=3)
    if f_img is not None:
        full_size_f = misc.to_tensor(full_size_f, dims=3)
    if c_img is not None:
        full_size_c = misc.to_tensor(full_size_c, dims=3)
    imgs = [full_size_img.unsqueeze(0).to(device, dtype=torch.float32),
            full_size_s.unsqueeze(0).to(device, dtype=torch.float32),
            full_size_t.unsqueeze(0).to(device, dtype=torch.float32)]
    if c_img is not None:
        imgs.append(full_size_c.unsqueeze(0).to(device, dtype=torch.float32))
    if f_img is not None:
        imgs.append(full_size_f.unsqueeze(0).to(device, dtype=torch.float32))
    img = img.unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        _, weights = net(img)
        if multi_scale:
            img_1 = resize(img, size=(int(0.5 * img.shape[2]), int(0.5 * img.shape[3])), interpolation='bilinear', align_corners=True)
            _, weights_1 = net(img_1)
            weights_1 = resize(weights_1, size=(img.shape[2], img.shape[3]), interpolation='bilinear', align_corners=True)

            img_2 = resize(img, size=(int(0.25 * img.shape[2]), int(0.25 * img.shape[3])), interpolation='bilinear', align_corners=True)
            _, weights_2 = net(img_2)
            weights_2 = resize(weights_2, size=(img.shape[2], img.shape[3]), interpolation='bilinear', align_corners=True)

            weights = (weights + weights_1 + weights_2) / 3
    weights = resize(weights, size=size, interpolation='bilinear', align_corners=True)

    if post_process:
        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                ref = imgs[0][j, :, :, :]
                curr_weight = weights[j, i, :, :]
                refined_weight = weight_refinement.process_image(ref, curr_weight, tensor=True)
                weights[j, i, :, :] = refined_weight
                weights = weights / torch.sum(weights, dim=1)

    for i in range(weights.shape[1]):
        if i == 0:
            out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
        else:
            out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
    out_img_np = out_img.squeeze().permute(1, 2, 0).cpu().numpy()
    return misc.outOfGamutClipping(out_img_np)
