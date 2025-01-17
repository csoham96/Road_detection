# Road_detection
Instructions to set up

1.Clone this repo
```bash
git clone https://github.com/csoham96/Road_detection.git
```

2.[Install CUDA](https://developer.nvidia.com/cuda-downloads)

3.Install pytorch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4.Install dependencies
```bash
pip install -r requirements.txt
```

5.Download Dataset from Kaggle (https://www.kaggle.com/datasets/eyantraiit/semantic-segmentation-datasets-of-indian-roads)

6.Training the model:

For training i have used the training set which consists around 2400 images,trained with th unet model for 30 epochs
Achieved a Dice score of 80.97%.
Specify the path of dataset inside train.py file
```bash
python train.py --amp -e 30
```

7.Converting into mobile-format run the file mobile_format.py which first converts model into torchscript and then into mobile format
```bash
python mobile_format.py
```

8.To infer on videos run infer_video.py 
```bash
python infer_video.py
```
