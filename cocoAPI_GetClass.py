from pycocotools.coco import COCO

ann_file = '/home/ian/下載/annotations/instances_train2014.json'
cocoGT = COCO(ann_file)

annIds = cocoGT.getAnnIds(iscrowd=None)
anns = cocoGT.loadAnns(annIds)

for i in range(0, len(anns)):
    print(anns[i]['category_id'])
