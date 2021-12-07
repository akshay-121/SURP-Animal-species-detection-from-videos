# SURP-Animal-species-detection-from-videos

<h1>
  Networks added:
 </h1>
  RCNN </br>
  FastRCNN </br>
  FasterRCNN </br>
  YOLO </br>

## How to run 

This repository contains well written algorithms to train a ML model to detect animal species in a image. We have trained it on around 11 animal species and weights corresponding to this is available here (https://drive.google.com/file/d/1-0fxwVB_4xXk--ZdZl-sA7nL05hmAqHI/view?usp=sharing). Number of classes we have trained are deliberately less as there is also an option to retrain the model on custom species. All we need is few annotated images of the particular aminal specie (annotation can be done by tool ((tool name)) ) then run the train script.
* Clone the repo
```
git clone  https://github.com/akshay-121/SURP-Animal-species-detection-from-videos.git
cd SURP-Animal-species-detection-from-videos
```
* Install requirements 
```
pip3 install -r requirements.txt
```

### Testing

```
python3 test.py --imagepath {imagepath} --mode test
```
where ```imagepath``` is the realtive path of image file to be tested.
It will classify and annotate the particular image based on the classes on which model is trained.

### Training 

## Future work
Currently it needs around 100 annotated images of the particular class for training to get good results. We are trying to make it learn more accurately even on 10s of images of a class. 
