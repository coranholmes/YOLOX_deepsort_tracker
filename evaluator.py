import os, argparse, json
import numpy as np
from utils.other import *
from torch import positive


def evaluate_exp(ds_name, mode):
    if mode == "frame":
        by_frame = True
    else:
        by_frame = False
    ds_root = os.path.join(os.getcwd(), "videos", ds_name)
    label_dir = os.path.join(ds_root, "label")
    for label in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label)
        gt_path = os.path.join(ds_root, "gt", label)
        process_label(label_path, gt_path, by_frame)


def process_label(label_path, gt_path, by_frame=True):
    print("Processing label file: %s" % label_path)
    label_file = open(label_path)
    gt_file = open(gt_path)

    gts = dict()
    for line in gt_file:
        # {"frame": 210, "type": "bus", "top": 212, "left": 117, "bottom": 343, "right": 286}
        d = json.loads(line)
        if d["frame"] in gts:
            gts[d["frame"]].append([d["top"], d["left"], d["bottom"], d["right"]])
        else:
            gts[d["frame"]] = [[d["top"], d["left"], d["bottom"], d["right"]]]

    positives = dict()
    p = 0
    for line in label_file:
        # {"frame": 210, "id": 6, "type": "bus", "top": 212, "left": 117, "bottom": 343, "right": 286, "parked_time": 5, "detected": "YES"}
        d = json.loads(line)
        if d["detected"] == "YES":  # TODO positives做预处理删除特别小的框
            p += 1
            if d["frame"] in positives:
                positives[d["frame"]].append(
                    [d["top"], d["left"], d["bottom"], d["right"]]
                )
            else:
                positives[d["frame"]] = [[d["top"], d["left"], d["bottom"], d["right"]]]
    print(gts)
    print(positives)
    tp, fn = 0, 0
    if (
        by_frame
    ):  # evaluate based on frames, every gt box in one frame is regarded as one positive sample
        for frame in gts:  # 遍历gt中的每个frame
            cur_frame_gt_cnt = len(gts[frame])
            match_cnt = 0
            for gt in gts[frame]:  #  遍历gt中的每个frame中的每个box
                if frame in positives:
                    for dec in positives[frame]:  # 对于对应frame中每个detection到的box进行匹配
                        iou = get_iou(gt, dec)
                        print(iou)
                        if iou > EVALUATION_IOU_THRESHOLD:
                            tp += 1
                            match_cnt += 1
            fn = fn + (cur_frame_gt_cnt - match_cnt)  # fn就是没匹配到的
    else:  # evaluate based on events
        pass

    fp = p - tp
    print("p:", p, "tp:", tp, "fp:", fp, "fn:", fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("f1:", f1)


def get_iou(gt, dec):
    gt = np.array(gt)
    dec = np.array(dec)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], dec[0])
    yA = max(gt[1], dec[1])
    xB = min(gt[2], dec[2])
    yB = min(gt[3], dec[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    boxAArea = (gt[2] - gt[0]) * (gt[3] - gt[1])
    boxBArea = (dec[2] - dec[0]) * (dec[3] - dec[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate results!")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="xd_full",
        help="ISLab|xd_full, choose the dataset to evaluate the experiment",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="frame",
        help="frame|event, choose the evaluation mode",
    )
    # parser.add_argument('-p', "--path", type=str, default="videos/ISLab/input/ISLab-04.mp4", help="choose a video to be processed")
    args = parser.parse_args()

    evaluate_exp(args.name, args.mode)
    # process_label("data/label.txt", "data/gt.txt")
