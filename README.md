# follow-me

You can run this from a folder with images (which correspond to video frames), or from a video. Also, you can skip detection phase by providing a detection file (MOT format).

## Create an ENV variable with the code path
```
export CODE_PATH=/Volumes/BACKUP_2TB/Maestria
```

## Run from images
```
python3 run.py --images_path=$CODE_PATH/Datasets/MOT16/test/MOT16-06/img1
```

## Run from video
```
python3 run.py --video_input=$CODE_PATH/Datasets/test.mp4
```

## Do not run detection phase and use detections from file
```
python3 run.py --images_path=$CODE_PATH/Datasets/MOT16/test/MOT16-06/img1 --load_detection_file=$CODE_PATH/Datasets/MOT16/test/MOT16-06/det/det.txt
```

## Run deep sort
```
python deep_sort_app.py --sequence_dir=../../../Datasets/MOT16/test/MOT16-06 --detection_file=$CODE_PATH/Datasets/MOT16/resources/detections/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True
```

## Remove useless files (causes conflics when reading images from folder)
```
cd $CODE_PATH/Datasets/MOT16/test
find . -type f -name '._*' -delete
```

## Run metrics (performance, accuracy) - Works with deep sort for now
Create tracking files
```
cd $CODE_PATH/Code/lib/deep_sort
python evaluate_motchallenge.py --mot_dir=$CODE_PATH/Datasets/MOT16/test --detection_dir=$CODE_PATH/Datasets/MOT16/resources/detections/MOT16_POI_test
```

Evaluate performance
```
python -m motmetrics.apps.eval_motchallenge $CODE_PATH/Datasets/MOT16/test $CODE_PATH/Code/lib/deep_sort/results --fmt mot16
```
