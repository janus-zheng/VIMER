''' autodocstring '''
import argparse

from PIL import Image
import paddle
from paddle.vision import transforms
from model.model_fusion import DeitBase


def process_image(args):
    """_summary_
    """
    model = DeitBase(args)  # args is not used in the class methods
    transform = transforms.Compose(
        [
            transforms.Resize(248),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    state_dict = paddle.load(args.model_weights)
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

    img = Image.open(args.input_image)
    img_tensor = transform(img).unsqueeze(axis=0)
    print(img_tensor.shape)
    fea_vec = model(img_tensor)
    print(fea_vec.shape)
    print(fea_vec)
    return 0


def main():
    """ main function """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-weights', type=str, required=True,
                        help='Path to the model file')
    parser.add_argument('--input-image', type=str, required=True,
                        help='Path to the input image')
    args = parser.parse_args()

    process_image(args)


if __name__ == '__main__':
    main()
