# follow-me

You can run this from a folder with images (which correspond to video frames), or from a video. Also, you can skip detection phase by providing a detection file (MOT format).

## Create an ENV variable with the code path
```
export CODE_PATH=/Volumes/128GB/Maestria
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
python deep_sort_app.py --sequence_dir=../../../Datasets/MOT16/test/MOT16-06 --detection_file=$CODE_PATH/Datasets/MOT16/resources_deep_sort/detections/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True
```

## Remove useless files (causes conflics when reading images from folder)
```
cd $CODE_PATH/Datasets/MOT16/train
find . -type f -name '._*' -delete
```

## Run metrics (performance, accuracy) - Works with deep sort for now

### Deep sort
Create tracking files
```
cd $CODE_PATH/Code/lib/deep_sort
python evaluate_motchallenge.py --mot_dir=$CODE_PATH/Datasets/MOT16/train --detection_dir=$CODE_PATH/Datasets/MOT16/resources_deep_sort/detections/MOT16_POI_train
```

Evaluate performance
Uses: `https://github.com/cheind/py-motmetrics`
```
python -m motmetrics.apps.eval_motchallenge $CODE_PATH/Datasets/MOT16/train $CODE_PATH/Code/lib/deep_sort/results --fmt mot16
```

Metrics using deep_sort MOT detections:
11:09:21 INFO - Found 7 groundtruths and 7 test files.
11:09:21 INFO - Available LAP solvers ['lap', 'scipy']
11:09:21 INFO - Default LAP solver 'lap'
11:09:21 INFO - Loading files.
11:09:25 INFO - Comparing MOT16-13...
11:09:32 INFO - Comparing MOT16-09...
11:09:35 INFO - Comparing MOT16-10...
11:09:41 INFO - Comparing MOT16-11...
11:09:46 INFO - Comparing MOT16-05...
11:09:49 INFO - Comparing MOT16-02...
11:09:58 INFO - Comparing MOT16-04...
11:10:32 INFO - Running metrics
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML    FP    FN IDs    FM  MOTA  MOTP
MOT16-13 56.1% 52.9% 59.8% 80.3% 71.1% 107  65  36  6  3745  2250 313   352 44.9% 0.237
MOT16-09 56.2% 64.3% 49.9% 71.5% 92.1%  25  13  11  1   322  1497  41    59 64.6% 0.161
MOT16-10 54.0% 53.4% 54.6% 76.4% 74.8%  54  24  29  1  3178  2905 242   310 48.7% 0.228
MOT16-11 63.3% 68.3% 58.9% 76.2% 88.4%  69  29  33  7   918  2187  65    96 65.4% 0.152
MOT16-05 60.8% 72.6% 52.3% 62.3% 86.4% 125  29  69 27   667  2573  65   115 51.5% 0.215
MOT16-02 40.7% 46.4% 36.2% 53.9% 69.1%  54  12  31 11  4302  8215 138   252 29.0% 0.210
MOT16-04 70.0% 76.0% 64.8% 71.9% 84.3%  83  42  26 15  6358 13358  71   255 58.4% 0.167
OVERALL  60.2% 64.4% 56.6% 70.1% 79.9% 517 214 235 68 19490 32985 935  1439 51.6% 0.189
11:23:50 INFO - Completed

### Follow me
```
python3 run.py --images_path=/Volumes/128GB/Maestria/Datasets/MOT16/train/MOT16-02/img1 --store_detections=results/MOT16-02.txt --no-display
```


python -m motmetrics.apps.eval_motchallenge $CODE_PATH/Datasets/MOT16/train2 /Volumes/128GB/Maestria/Code/results --fmt mot16
02:26:17 INFO - Found 1 groundtruths and 1 test files.
02:26:17 INFO - Available LAP solvers ['lap', 'scipy']
02:26:17 INFO - Default LAP solver 'lap'
02:26:17 INFO - Loading files.
02:26:17 INFO - Comparing MOT16-02...
02:26:24 INFO - Running metrics
          IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP    FN IDs   FM MOTA  MOTP
MOT16-02 27.1% 42.0% 20.0% 28.9% 60.6% 54  7 19 28 3343 12683  88  170 9.6% 0.269
OVERALL  27.1% 42.0% 20.0% 28.9% 60.6% 54  7 19 28 3343 12683  88  170 9.6% 0.269
02:26:40 INFO - Completed

