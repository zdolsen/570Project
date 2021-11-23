.RECIPEPREFIX +=

PYTHON=python3
ROOT=data/WIDER
TRAINDATA=$(ROOT)/wider_face_split/wider_face_train_bbx_gt.txt
VALDATA=$(ROOT)/wider_face_split/wider_face_val_bbx_gt.txt
TESTDATA=$(ROOT)/wider_face_split/wider_face_test_filelist.txt

CHECKPOINT=weights/checkpoint_10.pth
        
train:
        $(PYTHON) train.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT)

numworkers: 
        $(PYTHON) testNumWorkers.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT)

resume: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --epochs $(EPOCH)

evaluate: 
        $(PYTHON) evaluateWithPictures.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val

