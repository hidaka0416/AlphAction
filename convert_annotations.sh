#!/bin/bash

python tools/ava/csv2COCO.py --csv_path data/AVA/annotations/ava_train_v2.2.csv --movie_list data/AVA/annotations/ava_file_names_trainval_v2.1.txt --img_root data/AVA/keyframes/trainval
python tools/ava/csv2COCO.py --csv_path data/AVA/annotations/ava_val_v2.2.csv --movie_list data/AVA/annotations/ava_file_names_trainval_v2.1.txt --img_root data/AVA/keyframes/trainval
