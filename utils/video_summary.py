import cv2, argparse, os


def get_ds_length(ds_name):
    ds_root = os.path.join(os.getcwd(), 'videos', ds_name)
    input_dir = os.path.join(ds_root, 'input')
    video_list = [os.path.join(input_dir, vid_name) for vid_name in os.listdir(input_dir)]
    # print(video_list)
    duration = 0
    for vid_path in video_list:
        cap = cv2.VideoCapture(vid_path)    
        if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
            # get方法参数按顺序对应下表（从0开始编号)
            rate = cap.get(5)   # 帧速率
            FrameNumber = cap.get(7)  # 视频文件的帧数
            length = FrameNumber/rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
            print(vid_path, rate, FrameNumber, length)
            duration += length
    print("Total length: ", duration / 60, "min.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate the total length of a video dataset!")
    parser.add_argument('-n', "--name", type=str, default="ISLab", help="ISLab|xd_full, choose the dataset to run the experiment")
    args = parser.parse_args()
    get_ds_length(args.name)