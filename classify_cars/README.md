## Stanford cars dataset
Details are in https://ai.stanford.edu/~jkrause/cars/car_dataset.html

-The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split

- Here I use approach outlined in the fastai course.
- I use the fastai library in pytorch.
- I use transfer learning.
- Model is trained on Google Cloud Platform. (Machine type 4 vCPUs, 15 GB RAM, NVIDIA Tesla T4X1)
- First model is trained using weights from RESNET50. Each epoch takes about 3.5 minute to finish.

| Pretrain| #of epoch  | train_loss | valid_loss | accuracy |
| :---:   | :-: | :-: | :-: | :-:|
| ResNet50 | 10 | 0.086395 | 0.443805 | 0.884591 |


-- 
