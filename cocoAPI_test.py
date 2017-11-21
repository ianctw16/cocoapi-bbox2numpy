from pycocotools.coco import COCO
import numpy as np

ann_file = '/home/ian/下載/annotations/person_keypoints_val2014.json'
cocoGT = COCO(ann_file)

# getAnnIds(Give 0 imgIds return all annotations id)
annIds = cocoGT.getAnnIds(iscrowd=None)
anns = cocoGT.loadAnns(annIds)

'''
imgIds = cocoGT.getImgIds()
ids = cocoGT.loadImgs(imgIds)
print(ids)
'''

# print(anns[0])
d = {}
d2 = {}
d3 = {}

d.setdefault('boxes', [])
d2.setdefault('images', [])
d3.setdefault('ids', [])


for i in range(0, len(anns)):
    d['boxes'].append(anns[i]['bbox'])
for i in range(0, len(anns)):
    d2['images'].append(anns[i]['image_id'])
for id in d2['images']:
    d3['ids'].append(cocoGT.imgs[anns[i]['image_id']]['file_name'])


image_labels = [np.array(boxes, dtype=np.uint16) for boxes in d['boxes']]
image_labels = np.array(image_labels)
images = [np.array(id) for id in d3['ids']]
images = np.array(images)
print(images)
print(type(image_labels))
print(type(images))

np.savez("my_data", images=images, boxes=image_labels)

"""
data = np.load('my_data.npz')

image_data = data['images']
boxes = data['boxes']
print(image_data)
print(boxes)
"""
