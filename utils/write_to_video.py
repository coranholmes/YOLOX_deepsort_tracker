import cv2

if __name__ == '__main__':
    # vid = cv2.VideoCapture("videos/Sussex/input/SussexTrafficDay1.mpg")
    # if not vid.isOpened():
    #     raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter("videos/Sussex/output/SussexTrafficDay1.mp4", video_FourCC, 25, (720, 576)) 
    for i in range(2130):
        print(i)
        image = cv2.imread("videos/Sussex/capture/SussexTrafficDay1/" + str(i*25) + ".jpg")
        out.write(image)