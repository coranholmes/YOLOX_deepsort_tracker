import sys

sys.path.insert(0, "./YOLOX")
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch


class_names = COCO_CLASSES


class Tracker:
    def __init__(
        self,
        filter_class=None,
        model="yolox-s",
        ckpt="weights/yolox_s.pth.tar",
    ):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True,
        )
        self.filter_class = filter_class
        self.history = dict()

    def update(self, image):
        """
        image:  image frame needs to be tracked
        ts:    current timestamp (s)
        """
        info = self.detector.detect(image, visual=False)
        # print(info['class_ids'])
        outputs = []
        scores = []
        class_ids = []

        if info["box_nums"] > 0:
            bbox_xywh = []
            # bbox_xywh = torch.zeros((info['box_nums'], 4))
            for (x1, y1, x2, y2), class_id, score in zip(
                info["boxes"], info["class_ids"], info["scores"]
            ):
                if (
                    self.filter_class
                    and class_names[int(class_id)] not in self.filter_class
                ):
                    # print(class_names[int(class_id)] + "filtered!!!")
                    continue
                bbox_xywh.append(
                    [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
                )
                scores.append(score)
                class_ids.append(class_id)
            bbox_xywh = torch.Tensor(bbox_xywh)
            if (
                bbox_xywh.shape[0] != 0
            ):  # deal with the situation where no objectes are detected
                outputs = self.deepsort.update(bbox_xywh, scores, image)
        return outputs, scores, class_ids
