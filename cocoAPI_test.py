from pycocotools.coco import COCO
import numpy as np

ann_file = 'annotations/person_keypoints_val2014.json'
cocoGT = COCO(ann_file)

# getAnnIds(Give 0 imgIds return all annotations id)
annIds = cocoGT.getAnnIds(iscrowd=None)
anns = cocoGT.loadAnns(annIds)

# print(anns[0])
d = {}
d2 = {}
d.setdefault('boxes', [])
d2.setdefault('images', [])
test = np.array([])
print(type(test))
for i in range(0, len(anns)):
    d['boxes'].append(anns[i]['bbox'])
for i in range(0, len(anns)):
    d2['images'].append(anns[i]['image_id'])

image_labels = [np.array(boxes) for boxes in d['boxes']]
print(type(image_labels))
image_labels = np.array(image_labels)
print(type(image_labels))
