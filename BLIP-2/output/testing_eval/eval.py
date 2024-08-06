from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

import json
import os

def flickr30k_caption_eval(results_file):
    
    annotation_file = './f30k_butd_rand100_val_gt.json'
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)

    gt_img_ids = set(coco.getImgIds())
    print("GROUND_TRUTH:", gt_img_ids)
    print("=======================================")

    # Load results file
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract image IDs from results
    res_img_ids = set([item['image_id'] for item in results_data])
    print("RESULTS ID:", res_img_ids)
    print("***************************************")

    # Check if image IDs match
    if not res_img_ids.issubset(gt_img_ids):
        missing_ids = res_img_ids - gt_img_ids
        raise ValueError(f'The following image IDs in the results file are missing in the ground truth: {missing_ids}')
    
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

flickr30k_caption_eval('./val_epoch0.json')