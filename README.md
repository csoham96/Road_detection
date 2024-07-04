# Road_detection
Instructions to set up

1.Clone this repo

2.[Install CUDA](https://developer.nvidia.com/cuda-downloads)

3.Install pytorch

4.Install requirments.txt

5.Download Dataset from Kaggle (https://www.kaggle.com/datasets/eyantraiit/semantic-segmentation-datasets-of-indian-roads)

6.Training the model:
 
For training i have used the training set which consists around 2400 images,trained with th unet model for 30 epochs
Achieved a Dice score of 80.97% 

7.Converting into mobile-format run the file mobile_format.py which first converts model into torchscript and then into mobile format

8.To infer on videos run infer_video.py 
