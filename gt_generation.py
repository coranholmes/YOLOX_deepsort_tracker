import os, json, sys
from utils.other import *


def generate_labels(label_path, step, events):
    label_dir, label_name = os.path.split(label_path)
    ds_root = os.path.abspath(os.path.join(label_dir, ".."))
    gt_path = os.path.join("videos", ds_root, "gt", label_name)
    gt_file = open(gt_path, "w")
    if "ISLab" in label_path:
        id = int(label_path[-6:-4])
        frame_rate = ISLab_frame[id - 1]
    elif "ussex" in label_path:
        frame_rate = 25
    else:
        _, file_name = os.path.split(label_path)
        id = file_name[:-4]
        frame_rate = xd_full_frame[id]

    for id, event_start, event_end in events:
        matched = False
        event_start = event_start + (ILLEGAL_PARKED_THRESHOLD + N_INIT - 1) * frame_rate
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
            print(label_path + " does not detect!")
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

    # Label ISLab dataset
    # label_dir = os.path.join("videos", "ISLab", "label")
    # for file_name in os.listdir(label_dir):
    #     label_path = os.path.join(label_dir, file_name)
    #     id = int(label_path[-6:-4])
    #     print("processing " + label_path)
    #     generate_labels(label_path, ISLab_frame[id - 1], ISLab_label[id - 1])

    # Label xd_full dataset
    # label_dir = os.path.join("videos", "xd_full", "label")
    # for file_name in xd_full_lst:
    #     label_path = os.path.join(label_dir, file_name)
    #     id = file_name[:-4]
    #     generate_labels(label_path, xd_full_frame[id], xd_full_label[id])

    # Label Sussex dataset (Label single video example)
    label_dir = os.path.join("videos", "Sussex", "label")
    file_name = "SussexTrafficDay1.txt"
    label_path = os.path.join(label_dir, file_name)
    generate_labels(label_path, 25, sussex_label)
