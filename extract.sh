for image_size in 256 400 512; do
  for minimum in 5 10 20; do
    python surf_lbp.py -i /home/xandao/mygit/pr_dataset/GRAYSCALE/specific_epithet_trusted/${image_size}/${minimum} -o /home/xandao/pr_dataset_contraste/features/GRAYSCALE/${image_size}/${minimum} -c 1.2
  done
done