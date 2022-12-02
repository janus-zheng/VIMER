''' autodocstring '''
import io
import time
import argparse

import urllib3

import numpy as np
from PIL import Image
import paddle
from paddle.vision import transforms
from model.model_fusion import DeitBase


transform = transforms.Compose(
    [
        transforms.Resize(248),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def download_image_as_pillow_obj(u3_pool, url):
    """download image from a given url
    Args:
        u3_pool (urllib3 PoolManager instance): urllib3 PoolManager instance
        url (string): url of the images
    Returns:
        img (PIL image object): pillow image
    """

    try:
        req = u3_pool.request("GET", url)
    except urllib3.exceptions.TimeoutError as err:
        raise Exception(f"Invalid url: {url} with {err}") from err
    if req.status != 200:
        raise Exception(f"Status code error: {req.status},  url: {url}")
    img = None
    try:
        img = Image.open(io.BytesIO(req.data))
    except Exception as err:
        raise Exception(f"convert image content failed, error: {str(err)}") from err
    return img


def load_model_weights(model, weights):
    """ load model state dict """
    state_dict = paddle.load(weights)
    load_state_dict = {}
    for key, value in model.state_dict().items():
        key1 = key[6:]
        if key1 not in state_dict:
            print(f"{key1} is not found in modelpth")
        elif value.shape != state_dict[key1].shape:
            print(f"the shape {key} is unmatched: modelpath is {state_dict[key1].shape}, model is {value.shape}")
        else:
            load_state_dict[key] = state_dict[key1]
    model.set_state_dict(load_state_dict)
    model.eval()


def run_inference(model, img_pil):
    """ run inference on the pillow image, return a vector """
    img_tensor = transform(img_pil).unsqueeze(axis=0)
    since = time.time()
    fea_vec = model(img_tensor)
    print(f"inference time: {time.time() - since:.4f} seconds")
    return np.squeeze(fea_vec)


def process_image(args):
    """_summary_
    """
    model = DeitBase(args)  # args is not used in the class methods
    load_model_weights(model, args.model_weights)

    url1, url2 = args.input_urls
    u3_pool = urllib3.PoolManager(timeout=10)
    img_pil1 = download_image_as_pillow_obj(u3_pool, url1)
    img_pil2 = download_image_as_pillow_obj(u3_pool, url2)

    fea_vec1 = run_inference(model, img_pil1)
    fea_vec2 = run_inference(model, img_pil2)
    sim_score = np.dot(fea_vec1, fea_vec2)
    print(f"Similarity score: {sim_score:.4f}")

    return 0


def main():
    """ main function """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-weights', type=str, required=True,
                        help='Path to the model file')
    parser.add_argument('--input-urls', type=str, nargs=2, required=True,
                        help='URLs of the two input image')
    args = parser.parse_args()

    process_image(args)


if __name__ == '__main__':
    main()
