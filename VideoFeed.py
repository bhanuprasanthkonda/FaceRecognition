import cv2
from FaceRecognition import FaceRecognition


def VideoFeed():
    cam = cv2.VideoCapture(input("Enter the video file path: ").strip())
    fr = FaceRecognition()
    try:
        while True:
            successFullFrameRead, frame = cam.read()
            frame = fr.startUp(frame)
            cv2.imshow("Live Face Detection", frame)
            if cv2.waitKey(1) != -1:
                break
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()
        cam.release()


if __name__ == "__main__":
    VideoFeed()
