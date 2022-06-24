"""
SUDOKU SEGMENTATION TRAINING
"""
from detectron2.utils.logger import setup_logger
setup_logger()

# import numpy as np
import os, cv2, random
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import json
from detectron2.structures import BoxMode

def get_sudoku_dicts(directory):
    '''
    Transform labelme json to coco format

    Arguments:
         directory: contain polygon label of sudoku image in json format (generate with labelme)
    Returns:
         dataset_dicts: coco format used by detectron2
    '''
    classes = ['sudoku']
    dataset_dicts = []
    for idx, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# register label and image
for d in ["train", "test"]:
    DatasetCatalog.register("sudoku_" + d, lambda d=d: get_sudoku_dicts("../sudoku_img/detection_"+ d +"_label" ))
    MetadataCatalog.get("sudoku_" + d).set(thing_classes=['sudoku'])

#visualize training data

# my_dataset_train_metadata = MetadataCatalog.get("sudoku_train")
# dataset_dicts = DatasetCatalog.get("sudoku_train")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('image_show',vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows()

# configuration of the segmentation model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #load model
cfg.DATASETS.TRAIN = ("sudoku_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #load pretrain weight
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500 
cfg.SOLVER.STEPS = [] 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # 1 class for sudoku grid

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # create folder to saved new weight

#train model
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("sudoku_test", output_dir="./output")
test_loader = build_detection_test_loader(cfg, "sudoku_test")
print(inference_on_dataset(predictor.model, test_loader, evaluator))

