import os, json, sys

SUFFIX_LENGTH = 4  # the length of suffix (.mp4: length = 4)
ILLEGAL_PARKED_THRESHOLD = (
    5  # if the vehicle parks more than t(s), it will be marked as illegal
)
EVALUATION_IOU_THRESHOLD = (
    0.6  # minimum iou between detected vehicle and gt during evaluation
)

ILLEGAL_POCICY = "car"  # car|wheel|center
ILLEGAL_PARKING_MAX_RATIO = 0.3  # if the area of the vehicle / intersecion of vehicle and no-parking area > treshold, it is regarded as illegal parking behavior
ILLEGAL_PARKING_MAX_RATIO_W = 0.7  # similar to above but the RATIO is for wheel

MOVEMENT_RESTRICTION = True  # if the vehicle moves, counting would restart
MOVEMENT_MAX_IOU = 0.8  # the maximum iou between the location in the old and new frame of the same vehicle, ISLab=0.8, xd_full=0.9
N_INIT = 3  # deepsort parameter, tracker confirmed after N_INIT times


def get_exp_paras():
    name = ""
    policy = ILLEGAL_POCICY
    name = policy + "_"
    if policy == "car":
        name = name + str(ILLEGAL_PARKING_MAX_RATIO) + "_"
    elif policy == "wheel":
        name = name + str(ILLEGAL_PARKING_MAX_RATIO_W) + "_"
    elif policy == "center":
        name = name + "NA_"

    name = (
        name
        + str(MOVEMENT_RESTRICTION)
        + "_"
        + str(MOVEMENT_MAX_IOU)
        + "_"
        + str(EVALUATION_IOU_THRESHOLD)
    )
    return name


ISLab_frame = [20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
ISLab_label = [
    [(1, 0, 6240), (28, 180, 2020), (608, 4200, 6060), (346, 2260, 3740)],
    [(39, 520, 2000), (200, 2280, 3600), (344, 4160, 5600)],
    [(1, 0, 5400), (2, 0, 5400)],
    [(9, 2130, 4800)],
    [(6, 960, 4680)],
    [(23, 2370, 4650)],  # ISLab_06
    [(24, 1110, 3870)],
    [(60, 1080, 3120)],
    [(49, 1500, 3510)],
    [(67, 2130, 3570), (67, 3900, 4470)],
    [(64, 1530, 4110)],  # ISLab_11
    [(86, 1920, 3990)],
    [(15, 450, 2490)],
    [(1, 0, 8820), (134, 3660, 5250), (176, 5700, 8100)],  # ISLab_14
    [(6, 360, 3720)],
    [(8, 840, 2910)],
]

xd_full_lst = [
    "cloudy.txt",
    "night1.txt",
    "night2.txt",
    "rainy1.txt",
    "rainy2.txt",
    "sunny_shadow1.txt",
    "sunny_shadow2.txt",
    "sunny1.txt",
    "sunny2.txt",
]
xd_full_frame = {
    "cloudy": 25,
    "night1": 25,
    "night2": 30,
    "rainy1": 30,
    "rainy2": 30,
    "sunny_shadow1": 30,
    "sunny_shadow2": 30,
    "sunny1": 30,
    "sunny2": 30,
}
xd_full_label = {
    "cloudy": [(1, 125, 650), (19, 0, 1125), (8, 0, 3075), (327, 2075, 2575)],
    "night1": [],
    "night2": [],
    "rainy1": [(17, 0, 3210), (1818, 3540, 3660), (1852, 3570, 3720)],
    "rainy2": [],
    "sunny_shadow1": [],
    "sunny_shadow2": [(6, 0, 4110), (16, 0, 4110)],
    "sunny1": [(242, 360, 750)],
    "sunny2": [(762, 1980, 2160), (3, 0, 3990)],
}


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
