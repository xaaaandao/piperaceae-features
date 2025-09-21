#!/bin/bash

#for dataset in pr; do
#  for color in RGB; do
#    for min in "20"; do
#      for size in "256" "400" "512"; do
#        for model in "vgg16"; do
#          python main.py -i "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/original/images/${dataset}_dataset+20/${color}/${size}" -w ${size} -h ${size} -p 3 -m ${model} -s --orientation horizontal --output "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/original/features/${dataset}_dataset+20/"
#        done
#      done
#    done
#  done
#done


#for dataset in pr; do
#  for augmentation in Affine CLAHE GaussianBlur GridDistortion ISONoise    MotionBlur RandomGamma Blur GaussNoise HueSaturationValue MedianBlur OpticalDistortion  RandomRotate90 Transpose; do
#    for color in RGB; do
#      for min in "20"; do
#        for size in "256" "400" "512"; do
#          for model in "vgg16"; do
#            python main.py -i "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/data-augmentation/images/${dataset}_dataset+${min}+data-augmentation/${augmentation}/${color}/${size}" -w ${size} -h ${size} -p 3 -m ${model} -s --orientation horizontal --output "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/data-augmentation/features/${dataset}_dataset+${min}+data-augmentation/${augmentation}" -s
#          done
#        done
#      done
#    done
#  done
#done

for dataset in pr; do
  for augmentation in cut-mix; do
    for color in RGB; do
      for min in "20"; do
        for size in "256" "400" "512"; do
          for model in "vgg16"; do
            python main.py -i "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/data-augmentation/images/${dataset}_dataset+${min}+data-augmentation/${augmentation}/${color}/${size}" -w ${size} -h ${size} -p 3 -m ${model} -s --orientation horizontal --output "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/data-augmentation/features/${dataset}_dataset+${min}+data-augmentation/${augmentation}" -s
          done
        done
      done
    done
  done
done
