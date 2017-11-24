from pycocotools.coco import COCO
import numpy as np

f = open('valclass.txt', 'w')
ann_file = '/home/ian/下載/annotations/instances_val2014.json'
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
d_box = {}
d_images = {}
d_id = {}
d_class = []

d_box.setdefault('boxes', [])
d_images.setdefault('images', [])
d_id.setdefault('ids', [])

# store bbox in d['boxes']
# store category in d_class
for i in range(0, len(anns)):
    d_box['boxes'].append(anns[i]['bbox'])
    d_class.append(anns[i]['category_id'])

# store image_id in d_images['images']
for i in range(0, len(anns)):
    d_images['images'].append(anns[i]['image_id'])

# store file_name in d_id['ids']
for id in range(0, len(d_images['images'])):
    d_id['ids'].append(cocoGT.imgs[anns[id]['image_id']]['file_name'])


# trans d['boxes'] to numpy array
image_labels = [np.array(boxes, dtype=np.uint16) for boxes in d_box['boxes']]
image_labels = np.array(image_labels)
# trans d_id['ids'] --> file_name to numpy array
images = [np.array(id) for id in d_id['ids']]
images = np.array(images)
print(image_labels)
print(images)

# save files
np.savez("npzval2014", images=images, boxes=image_labels)
f.write(str(d_class))
f.close()

# data = np.load('my_data.npz')

'''read test
image_data = data['images']
boxes = data['boxes']
print(image_data)
print(boxes)
'''
