from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os, json
from multiprocessing import Process  # TODO add multiprocessing support
from utils.other import *
from glob import glob
from utils.visualize import vis_track
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES


class_names = COCO_CLASSES

def track_cap(vid_path):

    input_dir, vid_name= os.path.split(vid_path)
    ds_root = os.path.abspath(os.path.join(input_dir, ".."))
    
    capture_dir, label_dir, output_dir  = make_video_subdir(ds_root)
    capture_output_path = os.path.join(capture_dir, vid_name[:-SUFFIX_LENGTH])
    make_dir(capture_output_path)
    print(capture_output_path)

    # label_output_path = os.path.join(label_dir, vid_name[:-SUFFIX_LENGTH] + '.txt')
    # video_output_path = os.path.join(output_dir, "labelled_"+vid_name)
    
    vid = cv2.VideoCapture(vid_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # out = cv2.VideoWriter(video_output_path, video_FourCC, video_fps, video_size)  

    
    detector = Detector("yolox-s", "weights/yolox_s.pth.tar")

    idx = 0  # the idx of the frame
    image = None
    history = dict()  # the variable which stores tracking history: id -> timestamp first detected
    DETECT_EVERY_N_FRAMES = round(video_fps)  # detect every second

    while True:
        _, im = vid.read()
        if im is None:  # read finishes if there are no more frames
            break

        if idx % DETECT_EVERY_N_FRAMES == 0:
            # print(idx)
            # im = imutils.resize(im, height=500)
            info = detector.detect(im)
            image = info['visual']
            # save results to screen shots
            cv2.imwrite(os.path.join(capture_output_path, str(idx) + '.jpg'), image)

        # if image is not None:
        #     out.write(image)
        # else:
        #     out.write(im)
        
        idx += 1

    vid.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Detector Demo!")
    # parser.add_argument('-n', "--name", type=str, default="xd_full", help="ISLab|xd_full, choose the dataset to run the experiment")
    parser.add_argument('-p', "--path", type=str, default="videos/others/input/night1.mp4", help="choose a video to be processed")
    args = parser.parse_args()

    track_cap(args.path)

        