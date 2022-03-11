import sys, os, json
sys.path.insert(0, './YOLOX')
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from utils.visualize import vis_track


class_names = COCO_CLASSES
ILLEGAL_PARKED_THRESHOLD = 5  # if the vehicle parks more than t(s), it will be marked as illegal
file = open(os.path.join(os.getcwd(), 'res', 'text', "labels.txt"), 'w')  # TODO: change later

class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='weights/yolox_s.pth.tar', ):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class
        self.history = dict()

    def update(self, image, ts):
        """
        image:  image frame needs to be tracked
        ts:    current timestamp (s)
        """
        info = self.detector.detect(image, visual=False)
        outputs = []
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            text_labels = []
            json_texts = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for (x1, y1, x2, y2), class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                scores.append(score)
                cate = class_names[int(class_id)]
                text = cate + " " + str(round(score, 2))
                text_labels.append(text)

                                # write the time and location info to json file
                json_dict = {
                    'frame': ts,
                    'type': cate,
                    'top': int(y1),
                    'left': int(x1),
                    'bottom': int(y2),
                    'right': int(x2),
                }
                json_texts.append(json_dict)

            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, image)

            # Update parked time for each detected box
            for i in range(len(outputs)):
                box = outputs[i]
                id = box[4]
                if id not in self.history.keys():
                    self.history[id] = ts
                    text_labels[i] += " 0s"
                    parked_time = 0
                else:
                    parked_time = int(ts - self.history[id])
                    text_labels[i] = text_labels[i] + " " + str(parked_time) + "s"
                    if parked_time >= ILLEGAL_PARKED_THRESHOLD:  
                        text_labels[i] += " Detected!"
                
                json_texts[i]["parked_time"] = parked_time
                file.write(json.dumps(json_texts[i]) + "\n")

            image = vis_track(image, outputs, text_labels)

        return image, outputs
