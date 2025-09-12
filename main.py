import click
import numpy as np
import os
import pathlib
import tensorflow as tf

from dataset import Dataset
from image import Image
from model import get_model, get_input_shape
from patch import next_patch
from save import save_patches, save_features, save_samples

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


@click.command()
@click.option("--gpuid", type=int, default=0)
@click.option("-h", "--height", type=int, required=True)
@click.option("-i", "--input", required=True)
@click.option("-m", "--model", type=click.Choice(["mobilenetv2", "vgg16", "resnet50v2"]), required=True)
@click.option("--orientation", type=click.Choice(["horizontal", "vertical", "horizontal+vertical"]), required=True)
@click.option("-o", "--output", default="output")
@click.option("-p", "--patches", required=True, default=[1], multiple=True)
@click.option("-s", "--save_images", is_flag=True)
@click.option("-w", "--width", type=int, required=True)
def main(gpuid: int, height: int, input, model,
         orientation, output,
         patches: list, save_images: bool, width: int):

    print("Feature Extraction Parameters")
    print("Pre-trained model: %s" % model)
    print(f"Non-overlapping patches per image: {patches}")
    print("Orientation: %s" % orientation)
    print("Image Dimensions h=%d, w=%d" % (height, width))
    print("Format string for input: %s" % input)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    dataset = Dataset(height, input, model, orientation, width)
    images = []
    for patch in patches:
        print("Slicing patch_images into %d non-overlapping patches..." % patch)
        tf.keras.backend.clear_session()

        input_shape = get_input_shape(orientation, patch, height, width)
        model_name = model
        model, preprocess_input = get_model(model, weights="imagenet", include_top=False,
                                            input_shape=input_shape, pooling="avg")

        last_fold = -1
        features_shape = tuple()
        for idx, fold in enumerate(sorted(pathlib.Path(input).glob('*')), start=1):
            if fold.is_dir():
                features = []
                for filename in sorted(pathlib.Path(fold).glob("*.jpeg")):
                    patch_images = []
                    image = tf.keras.preprocessing.image.load_img(filename)
                    spec = tf.keras.preprocessing.image.img_to_array(image)
                    for p in next_patch(spec, patch, orientation):
                        p = preprocess_input(p)

                        # Armazena na lista a imagem recortada
                        if save_images:
                            patch_images.append(tf.keras.preprocessing.image.array_to_img(p))

                        p = np.expand_dims(p, axis=0)

                        # Armazena na lista as features extraídas
                        features_predict = model.predict(p)
                        features_predict = np.concatenate((features_predict, np.array([[idx]])), axis=1)
                        features_predict = np.concatenate((features_predict, np.array([[filename.stem]])), axis=1)
                        features.append(features_predict)

                    i = Image(filename, fold, patch_images)
                    images.append(i)
                features = np.concatenate(features)

                # idx = fold
                save_features(features, idx, model_name, output, patch)
                last_fold = idx
                features_shape = features.shape
        dataset.update(features_shape, last_fold, patch)
        dataset.save(output)

    if save_images:
        save_patches(images, output)
    save_samples(images, input, model_name, output)


if __name__ == '__main__':
    main()
