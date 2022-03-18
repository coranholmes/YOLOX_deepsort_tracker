from cProfile import label
import os, json, sys


def generate_labels(label_path, step, events):
    label_dir, label_name = os.path.split(label_path)
    ds_root = os.path.abspath(os.path.join(label_dir, ".."))
    gt_path = os.path.join("videos", ds_root, "gt", label_name)
    gt_file = open(gt_path, "w")
    for id, event_start, event_end in events:
        matched = False
        label_file = open(label_path)
        for line in label_file:
            json_dict = json.loads(line)
            if (
                event_start <= json_dict["frame"] <= event_end
                and json_dict["id"] == id
                and json_dict["detected"] == "YES"
            ):
                cate, y1, x1, y2, x2 = (
                    json_dict["type"],
                    json_dict["top"],
                    json_dict["left"],
                    json_dict["bottom"],
                    json_dict["right"],
                )
                matched = True
                print(line)
                break
        if not matched:
            print("label_path does not detect!")
            sys.exit(-1)

        for idx in range(event_start, event_end + 1, step):
            json_dict = {
                "frame": int(idx),
                "id": int(id),
                "type": cate,
                "top": int(y1),
                "left": int(x1),
                "bottom": int(y2),
                "right": int(x2),
            }
            gt_file.write(json.dumps(json_dict) + "\n")


if __name__ == "__main__":
    label_info = [
        [(1, 0, 6240), (28, 180, 2020), (614, 4200, 6060)],
        [(39, 520, 2000), (204, 2280, 3600), (349, 4160, 5600)],
        [(1, 0, 5400), (2, 0, 5400)],
        [(9, 2130, 4800)],
        [(6, 960, 4680)],
        [(23, 2370, 4650)],
        [(32, 1110, 3870)],
        [(72, 1080, 3120)],
        [(57, 1500, 3510)],
        [(71, 2130, 3570), (71, 3900, 4470)],
        [(69, 1530, 4110)],
        [(95, 1920, 3990)],
        [(15, 450, 2490)],
        [(1, 0, 8820), (157, 3660, 5250)],
        [(6, 360, 3720)],
        [(8, 840, 2910)],
    ]

    label_dir = os.path.join("videos", "ISLab", "label")
    for file_name in os.listdir(label_dir):
        label_path = os.path.join(label_dir, file_name)
        id = int(label_path[-6:-4])
        print("processing " + label_path)
        generate_labels(label_path, 30, label_info[id - 1])

    # ISLab
    # generate_labels(label_path, 30, [(1, 60, 6240), (28, 180, 2020), (614, 4200, 6060)])  # 1
    # generate_labels(label_path, 30, [(39, 520, 2000), (204, 2280, 3600), (349, 4160, 5600)])  # 2
    # generate_labels(label_path, 30, [(1, 60, 5400), (2, 60, 5400)])  # 3
    # generate_labels(label_path, 30, [(9, 2130, 4800)])  # 4
    # generate_labels(label_path, 30, [(6, 960, 4680)])  # 5
    # generate_labels(label_path, 30, [(23, 2370, 4650)])  # 6
    # generate_labels(label_path, 30, [(32, 1110, 3870)])  # 7
    # generate_labels(label_path, 30, [(72, 1080, 3120)])  # 8
    # generate_labels(label_path, 30, [(57, 1500, 3510)])  # 9
    # generate_labels(label_path, 30, [(71, 2130, 3570), (71, 3900, 4470)])  # 10
    # generate_labels(label_path, 30, [(69, 1530, 4110)])  # 11
    # generate_labels(label_path, 30, [(95, 1920, 3990)])  # 12
    # generate_labels(label_path, 30, [(15, 450, 2490)])  # 13
    # generate_labels(label_path, 30, [(1, 60, 8820), (157, 3660, 5250)])  # 14
    # generate_labels(label_path, 30, [(6, 360, 3720)])  # 15
    # generate_labels(label_path, 30, [(8, 840, 2910)])  # 16

    file_lst = [
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
    label_info = {
        "cloudy": [(19, 0, 1125), (8, 0, 3075), (327, 2075, 2575)],
        "night1": [],
        "night2": [],
        "rainy1": [(14, 0, 480), (318, 660, 1290), (17, 0, 3210)],
        "rainy2": [],
        "sunny_shadow1": [],
        "sunny_shadow2": [(6, 0, 4110), (16, 0, 4110)],
        "sunny1": [(15, 0, 2970), (2090, 4110, 4440), (768, 0, 4440), (847, 0, 4440)],
        "sunny2": [(766, 1980, 2160), (3, 0, 3990), (34, 0, 3990)],
    }
    # label_dir = os.path.join("videos", "xd_full", "label")
    # for file_name in file_lst:
    #     label_path = os.path.join(label_dir, file_name)
    #     id = file_name[:-4]
    #     print("processing " + label_path)
    #     generate_labels(label_path, 30, label_info[id])

    # xd_full
    # generate_labels(
    #    label_path,
    #     25,
    #     [(19, 0, 1125), (8, 0, 3075), (327, 2075, 2575)],
    # )  # cloudy
    # generate_labels(label_path, 25, [])  # night1
    # generate_labels(label_path, 25, [])  # night2
    # generate_labels(
    #     label_path, 25, [(14, 0, 480), (318, 660, 1290), (17, 0, 3210)]
    # )  # rainy1
    # generate_labels(label_path, 25, [])  # rainy2
    # generate_labels(label_path, 25, [])  # sunny_shadow1
    # generate_labels(label_path, 25, [(6, 0, 4110), (16, 0, 4110)])  # sunny_shadow2
    # generate_labels(
    #     label_path,
    #     25,
    #     [(15, 0, 2970), (2090, 4110, 4440), (768, 0, 4440), (847, 0, 4440)],
    # )  # sunny1
    # generate_labels(
    #     label_path,
    #     25,
    #     [(766, 1980, 2160), (3, 0, 3990), (34, 0, 3990)],
    # )  # sunny2
