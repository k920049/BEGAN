import os

import cv2
import numpy as np


class DataLoader:

    def __init__(self, path, extension = '.png'):
        self.path = path
        self.original_path = os.path.join(self.path, "img_align_celeba_png")
        self.cropped_path = os.path.join(self.path, "cropped")
        self.extension = extension

        self._load_data()

    def _load_data(self):
        # read the data from the cropped image
        im_files = os.listdir(self.cropped_path)
        # if there is no cropped data, then create them
        if len(im_files) == 0:
            self.crop_image()
        # again, read the image list
        im_files = os.listdir(self.cropped_path)
        images_plt = [cv2.imread(os.path.join(self.cropped_path, f)) for f in im_files if f.endswith(self.extension)]
        self.images = np.array(images_plt)

    def size(self):
        return self.images.shape[0]

    def crop_image(self):
        # read the images from the original data
        im_files = os.listdir(self.original_path)
        cascPath = './haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(cascPath)
        # preprocess every image
        for image_name in im_files:
            if image_name.endswith(self.extension):
                # read one image
                image = cv2.imread(os.path.join(self.original_path, image_name))
                image = np.array(image)
                # resize the image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 5, 5)

                if len(faces) == 0:
                    pass
                else:
                    x, y, w, h = faces[0]
                    image_crop = image[y:y + w, x:x + w, :]
                    image_resize = cv2.resize(image_crop, (64, 64))
                    print(image_resize.shape)
                    # image_converted = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(self.cropped_path, image_name), image_resize)

    def batch(self, batch_size):
        # select indices randomly for the images
        indices = np.random.randint(0, self.images.shape[0] - 1, batch_size)
        return self.images[indices]


