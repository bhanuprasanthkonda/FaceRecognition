import cv2
import time
import os
import warnings
warnings.filterwarnings("ignore")


def AddingNewFace():
    webCam = cv2.VideoCapture(0)
    newFaceName = input("Enter Name of the face: ").strip()
    if "trainingData" not in os.listdir():
        os.mkdir("trainingData")
    os.chdir("trainingData")
    if newFaceName not in os.listdir():
        os.mkdir(newFaceName)
    os.chdir(newFaceName)
    img_counter = len(os.listdir(os.curdir))
    while True:
        successFullFrameRead, frame = webCam.read()
        cv2.imshow("camera on (Press any key to stop)", frame)
        img_name = "{}.png".format(img_counter)
        print(img_name)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        time.sleep(1)
        img_counter += 1
        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            webCam.release()
            break

if __name__ == "__main__":
    AddingNewFace()
    print("Done")
