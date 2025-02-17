import os, json, sys, cv2, yaml
from skimage.metrics import structural_similarity

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
MOVEMENT_MAX_IOU = 0.9  # the maximum iou between the location in the old and new frame of the same vehicle, ISLab=0.8, xd_full=0.9
N_INIT = 3  # deepsort parameter, tracker confirmed after N_INIT times

SIMILARITY_RESTRICTION = True
SIMILARITY_THRESHOLD = 0.1
SIMILARITY_MIN_AREA = 1000  # bbox must be larger than the area so it can be processed by similarity strategy

TYPE_RESTRICTION = True
TYPE_MIN_FRAME = 1  # the type of the bbox must remains at least TYPE_MIN_FRAME frames, otherwise it might be a bad dection, ISLab = 5, xd_full = 1


def calc_similarity(img1, img2):
    if img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]:
        img1, img2 = img2, img1  # 保证old_crop比carnew_crop大
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # print(img1.shape, img2.shape)
    sim = structural_similarity(img1, img2, multichannel=True)
    return sim


def get_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


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
        + str(SIMILARITY_RESTRICTION)
        + "_"
        + str(SIMILARITY_THRESHOLD)
        + "_"
        + str(SIMILARITY_MIN_AREA)
        + "_"
        + str(EVALUATION_IOU_THRESHOLD)
        + "_"
        + str(TYPE_RESTRICTION)
        + "_"
        + str(TYPE_MIN_FRAME)
        + "__DS_"
    )

    config_file = "deep_sort/configs/deep_sort.yaml"
    with open(config_file, "r") as fo:
        cfg = yaml.safe_load(fo)
    cfg = cfg["DEEPSORT"]
    name = name + str(cfg["MAX_IOU_DISTANCE"]) + "_" + str(cfg["MAX_AGE"])
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

sussex_label = [
    (3928, 10050, 13900),
    (4547, 11350, 15100),
    (9947, 23700, 26725),
    (12262, 28550, 31700),
    (13600, 32350, 53225),
    (15401, 36075, 40725),
]

sussex_label2 = [
    (3928, 10050, 13900),
    (4547, 11350, 15100),
    (9947, 23700, 26725),
    (12262, 28550, 31700),
    (13600, 32350, 36800),
    (15934, 36825, 53225),
    (15401, 36075, 40725),
]


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
