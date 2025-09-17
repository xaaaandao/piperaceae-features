import dataclasses
import math

import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModel, ViTModel, ViTImageProcessor


@dataclasses.dataclass(init=False)
class Model:
    name: str
    gpu_id: int
    model = None
    preprocess_input = None
    feature_extractor = None

    def __init__(self, name, gpu_id):
        if name in ["vgg16", "resnet50v2", "mobilenetv2", "vit_huge", "vit_large", "vit_base", "vit_small"]:
            self.name = name
            self.gpu_id = gpu_id
        else:
            raise ValueError

    def set_model(self, **kwargs):
        if self.name in ["vgg16", "resnet50v2", "mobilenetv2"]:
            self.set_cnn(**kwargs)
        if self.name in ["vit_huge", "vit_large", "vit_base", "vit_small"]:
            self.set_transform()

    def set_transform(self):
        match self.name:
            case "vit_huge":
                self.model = ViTModel.from_pretrained("oogle/vit-huge-patch14-224-in21k").to(self.gpu_id)
                self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-huge-patch14-224-in21k")
            case "vit_large":
                self.model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k").to(self.gpu_id)
                self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
            case "vit_base":
                self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.gpu_id)
                self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            case "vit_small":
                self.model = AutoModel.from_pretrained("WinKawaks/vit-small-patch16-224").to(self.gpu_id)
                self.feature_extractor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
        self.model.eval()

    def set_cnn(self, **kwargs):
        match self.name:
            case "vgg16":
                self.model = tf.keras.applications.vgg16.VGG16(**kwargs)
                self.preprocess_input = tf.keras.applications.vgg16.preprocess_input
            case "resnet50v2":
                self.model = tf.keras.applications.resnet_v2.ResNet50V2(
                    **kwargs)
                self.preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
            case "mobilenetv2":
                self.model = tf.keras.applications.mobilenet_v2.MobileNetV2(**kwargs)
                self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    def cnn_features(self, filename, fold, patch):
        image_patch = self.preprocess_input(patch)
        p = np.expand_dims(image_patch, axis=0)

        # Adiciona o fold e o nome do arquivo
        features = self.model.predict(p)
        features = np.concatenate((features, np.array([[fold]])), axis=1)
        features = np.concatenate((features, np.array([[filename.stem]])), axis=1)
        return features, image_patch

    def transform_features(self, filename, fold, patch, save_images):
        inputs = self.feature_extractor(images=patch, return_tensors="pt")
        inputs = {key: value.to(self.gpu_id) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        return [last_hidden_state.mean(dim=1).squeeze().cpu().numpy()]

def get_input_shape(orientation: str, patch: int, spec_height: int, spec_width: int)-> tuple[int, int, int]:
    """
    Calcula o corte da imagem baseado na orientação.
    :param orientation: orientação do corte da imagem.
    :param patch: quantidade de divisões na imagem.
    :param spec_height: altura da imagem.
    :param spec_width: largura da imagem.
    :return: tupla com os valores de altura e largura da imagem.
    """
    match orientation:
        case "horizontal":
            return math.floor(spec_height / patch), spec_width, 3
        case "vertical":
            return spec_height, math.floor(spec_width / patch), 3

    raise ValueError