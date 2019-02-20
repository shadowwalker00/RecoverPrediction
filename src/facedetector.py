import cv2
import numpy as np
import matplotlib.pyplot as plt


class FaceDetector:
    """
    face detection class
    """
    def __init__(self):
        self.classifier = "./datasets/haarcascade_frontalface_default.xml"

    def detection(self, image):
        """
        :param image (numpy array)
        :return: the cropped image (numpy array)
        """
        face_cascade = cv2.CascadeClassifier(self.classifier)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces)>=1:
            areas = []
            for box in faces:
                (x, y, w, h)  = box
                areas.append(w*h)
            (x, y, w, h) = faces[areas.index(max(areas))]
            white_image = np.ones_like(image) * 255
            mask = np.zeros_like(image)
            center_x, center_y = (int((x + x + w) / 2), int((y + y + h) / 2))
            mask = cv2.ellipse(mask, center=(center_x, center_y), axes=(int(0.8*w), int(h)), angle=0, startAngle=0, endAngle=360,color=(255, 255, 255), thickness=-1)
            back_ground_mask = 255 - mask
            result = np.bitwise_and(image, mask)
            back_ground = np.bitwise_and(white_image,back_ground_mask)
            result += back_ground
            print(type(result))
            origin_y, origin_x,_ = result.shape
            y_start = max([0,center_y-int(h)])
            y_end = min([origin_y,center_y+int(h)])
            x_start = max([0,center_x-int(0.8*w)])
            x_end = min([origin_x,center_x+int(0.8*w)])
            result = result[y_start:y_end,x_start:x_end]
            #result = cv2.resize(result, (228, 256))
            return result,w,h
        else:
            return None


if __name__ == "__main__":
    detector = FaceDetector()
    image = cv2.imread("./test3.jpeg")
    try:
        plt.imshow(image)
        plt.show()
        re_shape = detector.detection(image)
        plt.figure()
        plt.imshow(re_shape)
        plt.show()
    except:
        print("Nothing")

