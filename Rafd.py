import numpy as np
from sklearn.model_selection import train_test_split
import glob, cv2

img_heigth = int(681/6)
img_width = int(1024/6)
folders = ["disgustado", "enojado", "feliz", "miedo", "neutral", "sorpresa", "triste"]

class Rafd:

    def __init__(self, path):
        self.path = path

    def extract_data(self):

        data = []
        labels = []

        for i, folder in enumerate(folders):
            files = glob.glob(self.path + folder + "/*.jpg")
            for myFile in files:
                image = cv2.imread(myFile)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                im_resized = cv2.resize(gray_image, (img_heigth, img_width), interpolation=cv2.INTER_LINEAR)
                data.append(im_resized)
                labels.append(i)

        return np.array(data), np.array(labels)

    def getData(self):

        features, labels = self.extract_data()
        features = features.astype('float32') / 255.
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        return x_train, x_test, y_train, y_test