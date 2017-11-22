from pycocotools.coco import COCO
import numpy as np

ann_file = '/home/ian/下載/annotations/person_keypoints_val2014.json'
cocoGT = COCO(ann_file)

# getAnnIds(Give 0 imgIds return all annotations id)
annIds = cocoGT.getAnnIds(iscrowd=None)
# anns has all annotation data
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

# store bbox in d['boxes']
for i in range(0, len(anns)):
    d['boxes'].append(anns[i]['bbox'])
# store image_id in d2['images']
for i in range(0, len(anns)):
    d2['images'].append(anns[i]['image_id'])
# store file_name in d3['ids']
for id in range(0, len(d2['images'])):
    d3['ids'].append(cocoGT.imgs[anns[id]['image_id']]['file_name'])

# trans d['boxes'] to numpy array
image_labels = [np.array(boxes, dtype=np.uint16) for boxes in d['boxes']]
image_labels = np.array(image_labels)
# trans d3['ids'] --> file_name to numpy array
images = [np.array(id) for id in d3['ids']]
images = np.array(images)
print(image_labels)
print(images)
np.savez("my_data", images=images, boxes=image_labels)

data = np.load('my_data.npz')

'''read test
image_data = data['images']
boxes = data['boxes']
print(image_data)
print(boxes)
'''
