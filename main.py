import os
import argparse
import torch
import gdown
import glog as log
from typing import List

import ips
from utils import io
from modeling import awb, restormer_arch

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


expected_landscape_img_height = 3464
expected_landscape_img_width = 5202
wb_model_name = "style-uformer_p_128_D_S_T_182"
wb_settings = wb_model_name.split("_")[3:-1]
wb_model_path = os.path.join("weights", wb_model_name + ".pth")
restormer_model_path = os.path.join("weights", "real_denoising.pth")


def single_run(
    base_dir: str,
    img_names: List,
    out_dir: str,
    wb_network,
    restormer,
    denoise_method: str = "bilateral",
    correction_method: str = "orj",
    autocontrast_method: str = "channel1"
):
    log.info(
        "Parameters:\n"
        f"Denoiser: {denoise_method}\n"
        f"Gamma Correction: {correction_method}\n"
        f"Auto-Contrast: {autocontrast_method}\n"
    )
    out_dir = out_dir.format(denoise_method, correction_method, autocontrast_method)
    for i, img_name in enumerate(img_names):
        p = round(100 * (i+1) / len(img_names), 2)
        log.info(f"({p:.2f}%) Processing {i+1} of {len(img_names)} images, image name: {img_name}")
        path = os.path.join(base_dir, img_name)
        assert os.path.exists(path)

        raw_image, metadata = io.read_image(path)
        metadata["exp_height"] = expected_landscape_img_height
        metadata["exp_width"] = expected_landscape_img_width
        metadata["wb_settings"] = wb_settings
        metadata["denoise_method"] = denoise_method
        metadata["correction_method"] = correction_method
        metadata["autocontrast_method"] = autocontrast_method
        
        out = ips.process(
            raw_image=raw_image,
            metadata=metadata,
            wb_network=wb_network,
            restormer=restormer
        )
        
        os.makedirs("./" + out_dir, exist_ok=True)
        out_path = os.path.join("./" + out_dir, img_name.replace("png", "jpg"))
        io.write_processed_as_jpg(out, out_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Night Photography Rendering Challenge - Team VGL OzU')
    parser.add_argument('-d', '--data_dir', type=str, default="data/", help="data directory")
    parser.add_argument('-w', '--weights_dir', type=str, default="weights/", help="weights directory")
    parser.add_argument('-o', '--output_dir', type=str, default="results/", help="output directory")
    parser.add_argument('-s', '--submission_name', type=str, default="vgl-ozu", help='submission name')
    args = parser.parse_args()
    
    weights_dir = args.weights_dir
    if not os.path.exists(weights_dir):
        log.info("Weights directory does not exist, then downloading the weights...")
        zip_fname = "weights.zip"
        io.download_weights("https://www.dropbox.com/s/5hvnbypzl86zyda/weights.zip?dl=1", zip_fname)
        os.makedirs(weights_dir)
        log.info("Unzipping downloaded zip file for weights...")
        io.unzip(zip_fname, weights_dir)
        os.remove(zip_fname)
        log.info("Done.")
        
    data_dir = args.data_dir
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        log.info("Data does not exist, please put the data from given link into 'data/test'...")
        os.makedirs(data_dir, exist_ok=True)
        log.info("After this, please re-run.")
    else:
        wb_network = awb.build_model(
            wb_model_path=wb_model_path,
            wb_settings=wb_settings, 
            device=io.get_device("0")
        )
        restormer = restormer_arch.build_model(
            model_path=restormer_model_path, 
            params= {
                'inp_channels':3, 
                'out_channels':3, 
                'dim':48, 
                'num_blocks':[4,6,6,8], 
                'num_refinement_blocks':4, 
                'heads':[1,2,4,8],
                'ffn_expansion_factor':2.66, 
                'bias':False, 
                'LayerNorm_type':'BiasFree', 
                'dual_pixel_task':False
            },
            device=io.get_device("1")
        ) if torch.cuda.device_count() > 1 else None
        
        base_dir = args.data_dir
        out_dir = args.output_dir
        img_names = os.listdir(base_dir)
        img_names = [img_name for img_name in img_names if ".png" in img_name]
        single_run(base_dir, img_names, out_dir, wb_network, restormer)