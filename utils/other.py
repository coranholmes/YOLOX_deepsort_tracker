import os, json, sys

SUFFIX_LENGTH = 4  # the length of suffix (.mp4: length = 4)
ILLEGAL_PARKED_THRESHOLD = (
    5  # if the vehicle parks more than t(s), it will be marked as illegal
)
EVALUATION_IOU_THRESHOLD = (
    0.8  # minimum iou between detected vehicle and gt during evaluation
)
ILLEGAL_PARKING_MAX_RATIO = 0.28  # if the area of the vehicle / intersecion of vehicle and no-parking area > treshold, it is regarded as illegal parking behavior

MOVEMENT_RESTRICTION = True  # if the vehicle moves, counting would restart
MOVEMENT_MAX_IOU = 0.9  # the maximum iou between the location in the old and new frame of the same vehicle


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


def get_mask_regions(mask_path, pic_name):
    print("Open mask file " + mask_path)
    mask_file = open(mask_path)
    mask_dict = json.load(mask_file)

    for i in range(len(mask_dict)):
        if mask_dict[i]["imageName"] == pic_name:
            masks = mask_dict[i]["Data"]
            mask_regions = []

            if len(masks) > 0:
                masks = masks["svgArr"]
                for poly in masks:
                    poly = poly["data"]
                    points = []
                    for p in poly:
                        points.append((p["x"], p["y"]))
                    mask_regions.append(points)
            return mask_regions
    print("Cannot find the mask regions!")
    sys.exit(-1)
