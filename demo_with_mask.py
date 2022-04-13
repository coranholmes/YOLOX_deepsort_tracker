from distutils.log import debug
from evaluator import get_iou
from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os, json, time
import numpy as np
from multiprocessing import Process
from utils.other import *
from glob import glob
from utils.visualize import vis_track
from utils.other import *
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from shapely.geometry import Polygon, Point


class_names = COCO_CLASSES

def multi_worker(args):
    vid_path, ds_name, show_masked, debugging = args.path, args.name, args.mask, args.debug
    if os.path.isfile(vid_path):  # process single video
        print("Processing single video path: %s" % vid_path)
        process_video(vid_path, show_masked, debugging)
    else:
        n_proc = 6
        ds_root = os.path.join(os.getcwd(), 'videos', ds_name)
        input_dir = os.path.join(ds_root, 'input')
        video_list = [os.path.join(input_dir, vid_name) for vid_name in os.listdir(input_dir)]

        chunks = [video_list[i::n_proc] for i in range(n_proc)]
        procs = []
        for chunk in chunks:
            if len(chunk) > 0:
                proc = Process(target=multi_process_video, args=(chunk, ), kwargs={"show_masked": show_masked, "debugging": debugging})
                procs.append(proc)
                proc.start()

        for proc in procs:
            proc.join()

def multi_process_video(chunk, show_masked, debugging):
    for vid_path in chunk:
        process_video(vid_path, show_masked, debugging)


def process_video(video_path, show_masked, debugging):

    input_dir, vid_name= os.path.split(video_path)
    ds_root = os.path.abspath(os.path.join(input_dir, ".."))
    
    print(ds_root)
    capture_dir, label_dir, output_dir  = make_video_subdir(ds_root)
    capture_output_path = os.path.join(capture_dir, vid_name[:-SUFFIX_LENGTH])
    label_output_path = os.path.join(label_dir, vid_name[:-SUFFIX_LENGTH] + '.txt')
    video_output_path = os.path.join(output_dir, vid_name)
    mask_path = os.path.join(ds_root, 'mask', 'mask.json')
    
    file = open(label_output_path, 'w')
    file_lines = []  # 暂存要写入label文件的内容
    dec_ids_set = set()  # 记录detect到的id
    mask_regions = get_mask_regions(mask_path, vid_name[:-SUFFIX_LENGTH] + ".jpg")

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # out = cv2.VideoWriter(video_output_path, video_FourCC, video_fps, video_size)  

    tracker = Tracker(filter_class=['car', 'bicycle', 'motorbike', 'bus', 'truck'], model="yolov3", ckpt="weights/yolox_darknet53.47.3.pth.tar")
    idx = 0  # the idx of the frame
    image = None
    history = dict()  
    DETECT_EVERY_N_FRAMES = round(video_fps)  # detect every second
   
    while True:
        # e1 = cv2.getTickCount()
        _, im = vid.read()
        if im is None:  # read finishes if there are no more frames
            break
        # idx += 1
        if idx % DETECT_EVERY_N_FRAMES == 0:
            # im = imutils.resize(im, height=500)
            ts = idx / DETECT_EVERY_N_FRAMES
            outputs, scores, class_ids = tracker.update(im)
            text_labels = []
            poly2 = None
            # Update parked time for each detected box
            for i in range(len(outputs)):
                box = outputs[i]
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                id = box[4]  # identifier of the detected object
                score = scores[i]
                class_id = class_ids[i]

                cate = class_names[int(class_id)]
                text = str(id) + " " + cate + " " + str(round(score, 2))

                parked_time = 0
                if id not in history.keys():
                    crop = im[y1:y2, x1:x2]
                    history[id] = {
                        "first_record":ts,  # 第一次被探测到时的timestamp
                        "region":(x1,y1,x2,y2),  # bbox的最新位置信息
                        "crop":crop,  # bbox的最新图像信息
                        "last_record":ts,  # 上一次被探测到时的timestamp
                        "type": cate,  # bbox的类别
                        "type_first_record": ts  # 记录到该类别的首次timestamp
                    }
                    parked_time = N_INIT - 1  # 第一次出现已经过去(N_INIT - 1)s
                else:
                    mask_clock = 1
                    movement_clock = 1

                    # filter those in masks
                    if ILLEGAL_POCICY == "car":
                        ## vanilla iou
                        poly1 = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])  # dtected bbox
                        for poly2 in mask_regions:
                            poly2 = Polygon(poly2)
                            intersection_area = poly1.intersection(poly2).area
                            a = poly1.area
                            if intersection_area / a <= ILLEGAL_PARKING_MAX_RATIO:
                                mask_clock = 0
                                break
                    elif ILLEGAL_POCICY == "wheel":
                        ## wheel iou
                        y1_prime = y1 + 2/3*(y2-y1)
                        poly1 = Polygon([(x1,y1_prime),(x1,y2),(x2,y2),(x2,y1_prime)])  # dtected bbox
                        for poly2 in mask_regions:
                            poly2 = Polygon(poly2)
                            intersection_area = poly1.intersection(poly2).area
                            a = poly1.area
                            if intersection_area / a <= ILLEGAL_PARKING_MAX_RATIO_W:
                                mask_clock = 0
                                break
                    elif ILLEGAL_POCICY == "center":
                        point = Point((x1+x2)/2, y2)
                        for poly2 in mask_regions:
                            poly2 = Polygon(poly2)
                            if not poly2.contains(point):
                                mask_clock = 0
                                break
                    
                    # add movement restriction
                    if mask_clock == 1 and MOVEMENT_RESTRICTION:
                        old_box = history[id]["region"]
                        iou = get_iou(old_box, (x1,y1,x2,y2))
                        if iou <= MOVEMENT_MAX_IOU: 
                            movement_clock = 0
                            # if id in [355]:
                            #     print(idx, id, iou, "movement clock restart!")

                    # add similarity restriction (只处理movement标为false的情况)
                    if mask_clock == 1 and movement_clock == 1 and get_area(x1,y1,x2,y2) >= SIMILARITY_MIN_AREA and SIMILARITY_RESTRICTION:
                        old_crop = history[id]["crop"]
                        new_crop = im[y1:y2, x1:x2]
                        sim = calc_similarity(old_crop, new_crop)
                        if sim < SIMILARITY_THRESHOLD and ts - history[id]["last_record"] >= 5:
                            movement_clock = 0
                    
                    # update type restriction info
                    old_cate = history[id]["type"]
                    if old_cate != cate:
                        history[id]["type_first_record"] = ts
                                                
                    # clock the parking time
                    if mask_clock == 0 or movement_clock == 0:
                        parked_time = 0  # 这里归零是因为在之前的策略中判定并非illegal所以重新计时
                        history[id]["first_record"] = ts   
                        history[id]["crop"] = im[y1:y2, x1:x2]
                    elif movement_clock == 1:
                        parked_time = int(ts - history[id]["first_record"])
                                
                    history[id]["region"] = (x1,y1,x2,y2)
                    history[id]["last_record"] = ts
                    history[id]["type"] = cate

                # e2 = cv2.getTickCount()
                # fps = cv2.getTickFrequency() / (e2 - e1)
                # print("FPS:", fps)

                
                # write the time and location info to json file
                json_dict = {
                    'frame': int(idx),
                    'id': int(id),
                    'type': cate,
                    'top': int(y1),
                    'left': int(x1),
                    'bottom': int(y2),
                    'right': int(x2),
                    'parked_time': int(parked_time),
                    'detected': 'YES' if parked_time >= ILLEGAL_PARKED_THRESHOLD else 'NO'
                }

                if TYPE_RESTRICTION and parked_time >= ILLEGAL_PARKED_THRESHOLD and ts - history[id]["type_first_record"] < TYPE_MIN_FRAME:
                        json_dict["detected"] = "TENTATIVE"
                if json_dict["detected"] == "YES":
                    dec_ids_set.add(json_dict["id"])
                
                # prepare labels for visualization
                text = text + " " + str(parked_time) + "s"
                if json_dict["detected"] == "YES":
                    text += " Detected!"
                elif json_dict["detected"] == 'TENTATIVE':
                    text += " Tentative"
                text_labels.append(text)
                file_lines.append(json_dict)

            if show_masked:  # if True, draw the masked area
                alpha = 0.3
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                overlay = im.copy()
                for poly2 in mask_regions:
                    poly2 = Polygon(poly2)
                    exterior = [int_coords(poly2.exterior.coords)]
                    cv2.fillPoly(overlay, exterior, color=(0, 255, 255))
                cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
            
            image = vis_track(im, outputs, text_labels, debugging)

            # save results to screen shots
            make_dir(capture_output_path)
            cv2.imwrite(os.path.join(capture_output_path, str(idx) + '.jpg'), image)

        # write results to a video
        # if image is not None:
        #     out.write(image)
        # else:
        #     out.write(im)
        
        idx += 1
    vid.release()

    for json_dict in file_lines:
        if json_dict["id"] in dec_ids_set and json_dict["detected"] == "TENTATIVE":
            json_dict["detected"] = "YES"
            json_dict["corrected"] = 1
        file.write(json.dumps(json_dict) + "\n")
    file.close()


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-n', "--name", type=str, default="ISLab", help="ISLab|xd_full, choose the dataset to run the experiment")
    parser.add_argument('-p', "--path", type=str, default="videos/ISLab/input/ISLab-13.mp44", help="choose a video to be processed")
    parser.add_argument('-m', '--mask', action="store_true", help="show masked area or not")   # default False， --mask changes the parameter to True
    parser.add_argument('-d', '--debug', action="store_true", help="show debugging results of deepsort tracking bbox")
    args = parser.parse_args()

    multi_worker(args)
    end = time.time()
    print("Processing time: ", end - start)

        