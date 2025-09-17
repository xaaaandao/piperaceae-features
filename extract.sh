#!/bin/bash

for dataset in pr; do
  for augmentation in Affine; do
    for color in RGB; do
      for min in "20"; do
        for size in "512"; do
          python main.py -i "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/images-redimensionadas-separada-por-especie/${dataset}_dataset+${min}+data-augmentation/${augmentation}/${color}/${size}" -w ${size} -h ${size} -p 3 -m vgg16 -s --orientation horizontal --output "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/features/${dataset}_dataset+${min}+data-augmentation/${augmentation}" -s
        done
      done
    done
  done
done