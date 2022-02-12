import cv2
import numpy as np
from imutils import paths


class FaceRecognition:
    def __init__(self):
        self.trainedFaceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.idName = None
        faces, ids = self.trainingData("trainingData")
        self.model = self.modelGenerator(faces,ids)

    def faceDetection(self, image):
        greyImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_coordinates = self.trainedFaceData.detectMultiScale(greyImg, scaleFactor=1.32, minNeighbors=5)
        return face_coordinates, greyImg

    def boxDrawer(self, img, face_coordinates):
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    def trainingData(self, path):
        from collections import defaultdict
        faces = []
        ids = []
        nameId = {}
        currentId = 1
        for imgPath in paths.list_images(path):
            name = imgPath.strip().split("/")[-2]
            if name in nameId:
                faceid = nameId[name]
            else:
                faceid = nameId[name] = currentId
                currentId += 1
            image = cv2.imread(imgPath)
            face_coordinate, greyImg = self.faceDetection(image)
            if len(face_coordinate) == 1:
                x, y, w, h = face_coordinate[0]
                greyImg = greyImg[y:y + h, x:x + w]
                faces.append(greyImg)
                ids.append(faceid)
        self.idName = {v: k for k, v in nameId.items()}
        return faces, ids

    def modelGenerator(self, faces, ids):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(faces, np.array(ids))
        return model

    def putText(self, image, text, x, y):
        cv2.putText(image, text, (x, y + 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    def startUp(self, img):
        face_coordinates, greyImg = self.faceDetection(img)
        print(face_coordinates)
        for face_coordinate in face_coordinates:
            x, y, h, w = face_coordinate
            face = greyImg[y:y + h, x:x + w]
            faceId, confidence = self.model.predict(face)
            print(faceId, confidence)
            self.boxDrawer(img, face_coordinates)
            self.putText(img, self.idName[faceId] if confidence < 70 else "Un-identified", x, y)
        return img


if __name__ == "__main__":
    img = cv2.imread('C:/Users/Bhanu Prasanth Konda/Desktop/9.png')
    img = FaceRecognition().startUp(img)
    cv2.imshow("photo (Press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done")
