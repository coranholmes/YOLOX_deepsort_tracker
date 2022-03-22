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
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from shapely.geometry import Polygon


class_names = COCO_CLASSES

def multi_worker(args):
    vid_path, ds_name, show_masked = args.path, args.name, args.mask
    if os.path.isfile(vid_path):  # process single video
        print("Processing single video path: %s" % vid_path)
        process_video(vid_path, show_masked)
    else:
        n_proc = 6
        ds_root = os.path.join(os.getcwd(), 'videos', ds_name)
        input_dir = os.path.join(ds_root, 'input')
        video_list = [os.path.join(input_dir, vid_name) for vid_name in os.listdir(input_dir)]

        chunks = [video_list[i::n_proc] for i in range(n_proc)]
        procs = []
        for chunk in chunks:
            if len(chunk) > 0:
                proc = Process(target=multi_process_video, args=(chunk, ), kwargs={"show_masked": show_masked})
                procs.append(proc)
                proc.start()

        for proc in procs:
            proc.join()

def multi_process_video(chunk, show_masked):
    for vid_path in chunk:
        process_video(vid_path, show_masked)


def process_video(video_path, show_masked):

    input_dir, vid_name= os.path.split(video_path)
    ds_root = os.path.abspath(os.path.join(input_dir, ".."))
    
    print(ds_root)
    capture_dir, label_dir, output_dir  = make_video_subdir(ds_root)
    capture_output_path = os.path.join(capture_dir, vid_name[:-SUFFIX_LENGTH])
    label_output_path = os.path.join(label_dir, vid_name[:-SUFFIX_LENGTH] + '.txt')
    video_output_path = os.path.join(output_dir, vid_name)
    mask_path = os.path.join(ds_root, 'mask', 'mask.json')
    
    file = open(label_output_path, 'w')
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
    history = dict()  # the variable which stores tracking history: id -> timestamp first detected
    DETECT_EVERY_N_FRAMES = round(video_fps)  # detect every second

    while True:
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
                    history[id] = [ts, (x1,y1,x2,y2)]
                    text += " 0s"

                else:
                    clock = True

                    # filter those in masks
                    poly1 = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])  
                    for poly2 in mask_regions:
                        poly2 = Polygon(poly2)
                        intersection_area = poly1.intersection(poly2).area
                        a = poly1.area
                        if intersection_area / a <= ILLEGAL_PARKING_MAX_RATIO:
                            clock = False
                    
                    # add movement restriction
                    if MOVEMENT_RESTRICTION:
                        old_box = history[id][1]
                        iou = get_iou(old_box, (x1,y1,x2,y2))
                        if iou <= MOVEMENT_MAX_IOU: 
                            clock = False
                    
                    # clock the parking time
                    if clock:
                        parked_time = int(ts - history[id][0])
                    else:
                        parked_time = 0
                        history[id][0] = ts                      
                    
                    history[id][1] = (x1,y1,x2,y2)
                    text = text + " " + str(parked_time) + "s"
                    if parked_time >= ILLEGAL_PARKED_THRESHOLD:  
                        text += " Detected!"
                text_labels.append(text)

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
                file.write(json.dumps(json_dict) + "\n")

            if show_masked:  # if True, draw the masked area
                alpha = 0.3
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                overlay = im.copy()
                for poly2 in mask_regions:
                    poly2 = Polygon(poly2)
                    exterior = [int_coords(poly2.exterior.coords)]
                    cv2.fillPoly(overlay, exterior, color=(0, 255, 255))
                cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
            
            image = vis_track(im, outputs, text_labels)

            # save results to screen shots
            make_dir(capture_output_path)
            cv2.imwrite(os.path.join(capture_output_path, str(idx) + '.jpg'), image)

        # if image is not None:
        #     out.write(image)
        # else:
        #     out.write(im)
        
        idx += 1

    file.close()
    vid.release()


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-n', "--name", type=str, default="ISLab", help="ISLab|xd_full, choose the dataset to run the experiment")
    parser.add_argument('-p', "--path", type=str, default="videos/ISLab/input/ISLab-13.mp44", help="choose a video to be processed")
    parser.add_argument('-m', '--mask', action="store_true", help="show masked area or not")   # default False， --mask changes the parameter to True
    args = parser.parse_args()

    multi_worker(args)
    end = time.time()
    print("Processing time: ", end - start)

        