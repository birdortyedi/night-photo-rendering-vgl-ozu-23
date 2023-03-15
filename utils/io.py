import cv2
import json
import torch
from pathlib import Path
from fractions import Fraction


def get_device(gpu_id=None):
    cuda_device = "cuda" 
    if gpu_id is not None:
        assert gpu_id in ["0", "1"]  # for local setup with 2 GPUs
        cuda_device += f":{gpu_id}"
    return torch.device(cuda_device if torch.cuda.is_available() else "cpu")


def fraction_from_json(json_object):
    if 'Fraction' in json_object:
        return Fraction(*json_object['Fraction'])
    return json_object


def json_read(fname, **kwargs):
    with open(fname) as j:
        data = json.load(j, **kwargs)
    return data


def read_image(path):
    png_path = Path(path)
    raw_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    metadata = json_read(png_path.with_suffix('.json'), object_hook=fraction_from_json)
    return raw_image, metadata


def write_processed_as_jpg(out, dst_path, quality=100):
    cv2.imwrite(dst_path, out, [cv2.IMWRITE_JPEG_QUALITY, quality])


def download_weights(url, fname):
    import requests
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
                f.flush()
                
                
def unzip(path_to_zip_file, directory_to_extract_to):
    import zipfile
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)