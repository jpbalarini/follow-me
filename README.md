# follow-me

You can run this from a folder with images (which correspond to video frames), or from a video. Also, you can skip detection phase by providing a detection file (MOT format).

## Run from images
python3 run.py --images_path=/Volumes/BACKUP_2TB/Maestria/Datasets/MOT16/test/MOT16-06/img1

## Run from video
python3 run.py --video_input=/Volumes/BACKUP_2TB/Maestria/Datasets/test.mp4

## Do not run detection phase and use detections from file
python3 run.py --images_path=/Volumes/BACKUP_2TB/Maestria/Datasets/MOT16/test/MOT16-06/img1 --load_detection_file=/Volumes/BACKUP_2TB/Maestria/Datasets/MOT16/test/MOT16-06/det/det.txt

## Run deep sort
python deep_sort_app.py --sequence_dir=../../../Datasets/MOT16/test/MOT16-06 --detection_file=/Volumes/BACKUP_2TB/Maestria/Datasets/MOT16/resources/detections/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True

## Remove useless files (causes conflics when reading images from folder)
/Volumes/BACKUP_2TB/Maestria/Datasets/MOT16/test
find . -type f -name '._*' -delete
