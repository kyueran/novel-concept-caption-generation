import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.data import MetadataCatalog
import cv2
import os
import numpy as np

# Define constants
NUM_OBJECTS = 36  # Number of objects to detect
CONFIG_PATH = "./VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"  # Replace with your config path
MODEL_WEIGHTS_PATH = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"  # Replace with your model weights path
IMAGE_PATH = "./65567.jpg"  # Replace with your image path

def verify_paths(config_path, image_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")

# Load the Detectron2 model
def load_model(config_path, model_weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_weights_path
    predictor = DefaultPredictor(cfg)
    return predictor

# Extract features from an image
def extract_features(predictor, image_path):
    im = cv2.imread(image_path)
    raw_height, raw_width = im.shape[:2]
    
    with torch.no_grad():
        image = predictor.transform_gen.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": raw_height, "width": raw_width}
        
        images = predictor.model.preprocess_image([inputs])
        features = predictor.model.backbone(images.tensor)
        
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])
        
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        
        # Use the output predictions and proposals for inference
        pred_scores = pred_class_logits.softmax(dim=-1)
        
        # Select the proposals with highest scores
        num_instances = min(len(pred_scores), NUM_OBJECTS)
        topk_indices = pred_scores[:, 1:].max(dim=1)[0].topk(num_instances).indices
        
        instances = Instances((raw_height, raw_width))
        instances.pred_boxes = proposal_boxes[0][topk_indices]
        instances.scores = pred_scores[topk_indices]
        instances.pred_classes = pred_scores[topk_indices].argmax(dim=1)
        
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[topk_indices].detach()
        
        return instances, roi_features

# Main function to run the feature extraction
def main():
    # Verify paths before loading the model
    verify_paths(CONFIG_PATH, IMAGE_PATH)
    
    predictor = load_model(CONFIG_PATH, MODEL_WEIGHTS_PATH)
    instances, features = extract_features(predictor, IMAGE_PATH)
    print("Instances:", instances)
    print("Extracted features shape:", features.shape)

if __name__ == "__main__":
    main()