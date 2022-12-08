# forklift-human-tracker-yolov5
This repository is contribution for YOLOv5.

Final video YouTube link: https://youtu.be/Pp9YI-KdhUQ 


# REQUIREMENTS
First of all, what is YOLOv5?

YOLO stands for "You Only Look Once" and is an extremely fast object detection framework using a single convolutional network. YOLO is frequently faster than other object detection systems because it looks at the entire image at once as opposed to scanning it pixel-by-pixel. YOLO does this by breaking an image into a grid, and then each section of the grid is classified and localized (i.e. the objects and structures are established). Then, it predicts where to place bounding boxes. Predicting these bounding boxes is done with regression-based algorithms, as opposed to a classification-based one.

Generally, classification-based algorithms are completed in two steps: first, selecting the Region Of Interest (ROI), then applying the convolutional neural network (CNN) to the regions selected to detect object(s).

YOLO's regression algorithm predicts the bounding boxes for the whole image at once, which is what makes it dramatically faster and a great option to boot.

You'll need these techs in your computer before starting off: (Anaconda stuff is so slow and using it with these techs makes things got messy. So we'll do the other, straight, clean way.)

-YOLOv5 repository which you'll find here: https://github.com/ultralytics/yolov5
-Python 3.8 or later
-PyTorch
-CUDA (

CUDA (or Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs (GPGPU). CUDA is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of compute kernels. In short, CUDA is an architecture and technology for GPU that is available as a plug-in on NVIDIA's C programming language.

Note that: CUDA is only supported on Nvidia Quadro and Geforce 8, 9, 200 series and all newer cards.

)

-You can pull Yolo with "git clone" or work as it is in Google Colab. I'm going to show you how you can pull it into your local computer and start working right away.

First, you can download the repository as you can see below:
![yolo1](https://user-images.githubusercontent.com/71467992/206456979-f7e146f5-f219-4224-9710-94bf60ecd2f4.png)

Then make sure you've got Python 3.8 or later. (!!Considerable: You should consider downloading Python 3.9 version for example. Because NVIDIA CUDA does not work with the latest Python versions now. Please consider previous stable versions. If you do otherwise, you will encounter errors. You can do trial-error. And these stuff can change soon of course.)

After you've downloaded the right version of Python. You MUST see these on your PATH: (So please select "Add to Path" option while downloading it)

![python ortamlar](https://user-images.githubusercontent.com/71467992/206459810-a6ab3c87-aae5-4aa6-9afd-55ad816c892f.png)

After doing this too, you can download and install CUDA like this: (for Windows)

1- Go https://developer.nvidia.com/cuda-downloads

2- ![cuda2](https://user-images.githubusercontent.com/71467992/206460771-ac587884-a006-4568-9288-e32700f512ea.png)

3- And open the installer ant let it install CUDA fastly on your system step by step.

After you make sure that CUDA is okay too, you can proceed to PyThorch stage.

-In this stage, go to https://pytorch.org/ and follow these steps: 

![pytorch](https://user-images.githubusercontent.com/71467992/206464027-44b45e80-13d6-45d7-9d10-d80b0ff0441b.png)

Then, in your C:\Users\YourUserName\AppData\Local\Programs\Python\Python39\Scripts directory, open the command line in it and run the command that you've copied from the Torch's website. 

![cmd](https://user-images.githubusercontent.com/71467992/206466776-c60c37c2-663c-468c-ad42-032926828ae9.png)

By the way, Torch works fine with the 11.8 (the latest version of CUDA). And in case you don't have a NVIDIA GPU, then you can select the CPU option while downloading PyTorch.

-In here, it might want you to update "pip" version. You can do this typing this command --> python -m pip install --upgrade pip and you can check your pytorch version running these commands:
---import torch 
---print(torch.__version__)

# IN YOLO
-And, we finally made it to YOLOv5 directory step!

-Go to YOLOv5 directory wherever you did the downloading into. And open the command line in it again.

1- Run this --> pip install -r requirements.txt for additional requirements that YOLO required.

2- It might want you to download Visual Studio. If you don't have, you must download and install it.

3- Run this --> pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI for cocoAPI

4- If you would like to capture your cam, run this --> python detect.py --source 0 (if you have only one cam, otherwise change the index at the end.)

5- To work with a video in your local, do these:

--- Download and copy this dataset into your yolov5-master directory: https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset
--- In ../yolov5-master/data/coco128.yaml file, change the directories like this:

path: C:\Users\YourUserName\Desktop\yolov5-master\data\dataset\images 
train: C:\Users\YourUserName\Desktop\yolov5-master\data\dataset\images\train  
val: C:\Users\YourUserName\Desktop\yolov5-master\data\dataset\images\val  
test:  C:\Users\YourUserName\Desktop\yolov5-master\data\dataset\images\test (optional)

--- In the same file, under the paths, change the first two class name like this:

names:
  0: forklift
  1: person
  
--- Save the file (maybe as mycustomdata.yaml)

--- In ../yolov5-master/detect.py file, go to "save_img" function and change the suffix like that:

![xdasdas](https://user-images.githubusercontent.com/71467992/206473334-e839ffd9-45f3-4d3a-9ce6-0c315341c9f7.png)

--- In the same file, type CTRL + F and search for "waitKey". Find it and switch it to "cv2.waitKey(10). (or 15-20 maybe. it's the video speed)
--- In ../yolov5-master/utils/plots.py file, search for "Colors" class and find it. In it, change the bounding box color as you wish:

![colors](https://user-images.githubusercontent.com/71467992/206475887-e5c7f002-8266-45fa-b9a6-f2a3838a3b91.png)

--- In the detect.py, go to line 242 and change the default value of --line-thickness as "1"
--- In plots.py, go to "box_label" file and add the following part in here:

![Screenshot_12](https://user-images.githubusercontent.com/71467992/206481505-2a50fac4-a51d-452c-88cb-b6c6861bc7fa.png)

--- We can now train our entity

- Run this --> python train.py --img 640 --batch 16 --epochs 5 --data yourcustomdata.yaml --weights yolov5s.pt
- After training, run this --> python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source ../yourvideo.mp4 --view-img 
- View it and wait till the end so it can scan it.


--- Now go to /yolov5-master/runs/detect/exp and check if the final video is there.


-AND YOU'RE DONE. THAT'S IT. KEEP GOING!
















