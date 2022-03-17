from cProfile import label
import os, json, sys


def generate_labels(label_path, step, events):
    label_dir, label_name = os.path.split(label_path)
    ds_root = os.path.abspath(os.path.join(label_dir, ".."))
    label_file = open(label_path)
    gt_path = os.path.join("videos", ds_root, "gt", label_name)
    gt_file = open(gt_path, "w")
    for id, event_start, event_end in events:
        print(id, event_start, event_end)
        matched = False
        for line in label_file:
            json_dict = json.loads(line)
            if json_dict["frame"] == event_start:
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
        [(1, 60, 6240), (28, 180, 2020), (614, 4200, 6060)],
        [(39, 520, 2000), (204, 2280, 3600), (349, 4160, 5600)],
        [(1, 60, 5400), (2, 60, 5400)],
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
        [(1, 60, 8820), (157, 3660, 5250)],
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
