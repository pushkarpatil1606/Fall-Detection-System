# Milestone 1 dataset setup

This package prepares the existing D:\Fall-Detection-Data dataset for
Ultralytics without copying the large COCO/VOC image files.

Run from the project root with the `.venv` active:

```powershell
python scripts/create_detection_dataset.py
```

Then verify:

```powershell
python scripts/verify_detection_dataset.py
```

The generated dataset is:

```text
D:\Fall-Detection-Data\processed\detection\yolo\
├── train\
│   ├── images\
│   └── labels\
└── val\
    ├── images\
    └── labels\
```

Train:
- COCO train2017 person images
- VOC2007 train split person images

Validation:
- COCO val2017 person images

`configs/detection_person.yaml` points Ultralytics to this generated dataset.

The images are hard-linked, not copied. Labels are small copied files with
`coco_` and `voc2007_` prefixes to keep image/label basenames unique.
