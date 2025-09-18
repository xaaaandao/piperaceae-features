import click
import numpy as np
import os
import pathlib
import tensorflow as tf
import torch

from dataset import Dataset
from image import Image
from patch import next_patch
from save import save_patches, save_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


@click.command()
@click.option("-c", "--color", type=str, default="RGB")
@click.option("--gpuid", type=int, default=0)
@click.option("-h", "--height", type=int, required=True)
@click.option("-i", "--input", required=True)
@click.option("-m", "--model", type=click.Choice(["mobilenetv2", "vgg16", "resnet50v2", "vit_huge", "vit_large", "vit_base", "vit_small"]), required=True)
@click.option("--orientation", type=click.Choice(["horizontal", "vertical", "horizontal+vertical"]), required=True)
@click.option("-o", "--output", default="output")
@click.option("-p", "--patch", required=True, default=1)
@click.option("-s", "--save_images", is_flag=True, default=True)
@click.option("-w", "--width", type=int, required=True)
def main(color: str, gpuid: int, height: int, input, model,
         orientation, output,
         patch: int, save_images: bool, width: int):

    print("Feature Extraction Parameters")
    print("Pre-trained model: %s" % model)
    print(f"Non-overlapping patches per image: %d" % patch)
    print("Orientation: %s" % orientation)
    print("Image Dimensions h=%d, w=%d" % (height, width))
    print("Format string for input: %s" % input)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    dataset = Dataset(color, gpuid, height, input, model, orientation, patch, width)

    print("Slicing patch_images into %d non-overlapping patches..." % patch)
    tf.keras.backend.clear_session()

    features_shape = tuple()
    last_fold = -1
    for idx, fold in enumerate(sorted(pathlib.Path(input).glob('*')), start=1):
        if fold.is_dir():
            features = []
            for filename in sorted(pathlib.Path(fold).glob("*.jpeg")):
                patch_images = []
                image = tf.keras.preprocessing.image.load_img(filename)
                spec = tf.keras.preprocessing.image.img_to_array(image)
                for p in next_patch(spec, patch, orientation):
                    if dataset.model.name in ["vgg16", "mobilenetv2", "resnet50v2"]:
                        f, image_patch = dataset.model.cnn_features(filename, idx, p)

                    if dataset.model.name in ["vit_huge", "vit_large", "vit_base", "vit_small"]:
                        f = dataset.model.transform_features(filename, idx, tf.keras.preprocessing.image.array_to_img(p), save_images)

                    patch_images.append(tf.keras.preprocessing.image.array_to_img(image_patch))
                    features.append(f)

                i = Image(filename, idx, patch_images, filename.parent.name)
                dataset.images.append(i)

            features = np.concatenate(features)

            # idx = fold
            save_features(dataset, features, idx, dataset.model.name, output, patch)

            last_fold = idx
            features_shape = features.shape
        dataset.update(features_shape, last_fold, patch)
        dataset.save(output)

    dataset.save_patches(output)
    dataset.save_samples(input, output)


if __name__ == '__main__':
    main()
