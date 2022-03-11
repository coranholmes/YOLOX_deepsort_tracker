from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob



def track_images(img_dir):
    tracker = Tracker(model='yolox-s', ckpt='weights/yolox_s.pth.tar',filter_class=['truck','person','car'])
    imgs = glob(os.path.join(img_dir,'*.png')) + glob(os.path.join(img_dir,'*.jpg')) + glob(os.path.join(img_dir,'*.jpeg'))
    for path in imgs:
        im = cv2.imread(path)
        im = imutils.resize(im, height=400)
        image,_ = tracker.update(im)
        #image = imutils.resize(image, height=500)

        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cv2.destroyAllWindows()



def track_cap(file, dest_dir, mode):
    vid = cv2.VideoCapture(file)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    DETECT_EVERY_N_FRAMES = round(video_fps)  # detect every second
    out = cv2.VideoWriter(os.path.join(dest_dir, "res.mp4"), video_FourCC, video_fps, video_size)

    tracker = Tracker(filter_class=['car', 'bicycle', 'motorbike', 'bus', 'truck'])
    idx = 0
    image = None
    while True:
        
        _, im = vid.read()
        if im is None:
            break
        idx += 1
        if idx % DETECT_EVERY_N_FRAMES == 0:
            # im = imutils.resize(im, height=500)
            image,_ = tracker.update(im, idx / DETECT_EVERY_N_FRAMES)
        
            # show the video result in real time
            # cv2.imshow('demo', image)
            # cv2.waitKey(1)
            # if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            #     break

            # save results to screen shots
            cv2.imwrite(os.path.join(os.getcwd(), 'res', str(idx) + '.jpg'), image)

        if mode == "all":
            # save results to a video
            if image is not None:
                out.write(image)
            else:
                out.write(im)

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, default='data/ISLab-11.mp4', help="choose a video to be processed")
    parser.add_argument('-r', "--respath", type=str, default='res', help="choose a destination path to store result")
    parser.add_argument('-m', "--mode", type=str, default='all', help="img|all, output image captures only or both images and videos")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        track_cap(args.path, args.respath, args.mode)
    else:
        track_images(args.path)
        