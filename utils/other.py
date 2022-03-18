import os

SUFFIX_LENGTH = 4  # the length of suffix (.mp4: length = 4)
ILLEGAL_PARKED_THRESHOLD = (
    5  # if the vehicle parks more than t(s), it will be marked as illegal
)
EVALUATION_IOU_THRESHOLD = 0.8  # minimum iou during evaluation


def make_dir(path):
    if not os.path.exists(path):
        print("Creating path {}".format(path))
        os.mkdir(path)


def make_video_subdir(ds_root):
    capture_output_path = os.path.join(ds_root, "capture")
    make_dir(capture_output_path)
    text_output_path = os.path.join(ds_root, "label")
    make_dir(text_output_path)
    video_output_path = os.path.join(ds_root, "output")
    make_dir(video_output_path)
    return capture_output_path, text_output_path, video_output_path
