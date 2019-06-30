# Mask_RCNN_tf
This is a tensorflow program of Mask_RCNN

you should create three folders before train: "coco", "checkpoint" and "logs"

put you coco dataset in "coco" folder, it has three sub folder: train2014, val2014 and annotations

the model will be saved in "checkpoint" folder and the training log will be saved in "logs".

# demo
put checkpoint files in "checkpoint" folder

run "python demo.py"

it will get image in "images" folder and predict

# train
prepare training dataset as readme file in coco

run "python coco_train.py"
