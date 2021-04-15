import cv2
from FaceRecognition import FaceRecognition


def liveVideoFeed():
    cam = cv2.VideoCapture(0)
    fr = FaceRecognition()
    try:
        while True:
            successFullFrameRead, frame = cam.read()
            frame = fr.startUp(cv2.flip(frame, 1))
            cv2.imshow("Live Face Detection", frame)
            if cv2.waitKey(1) != -1:
                break
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()
        cam.release()


if __name__ == "__main__":
    liveVideoFeed()
