import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 480
mountain = cv2.resize(mountain, (640, 480))

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # creating thresholds for skin color (example values, you may need to tune them)
        lower_bound = np.array([0, 20, 70], dtype=np.uint8)
        upper_bound = np.array([20, 255, 255], dtype=np.uint8)

        # thresholding image to get binary mask for skin color
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask to get the background (mountain) area
        inv_mask = cv2.bitwise_not(mask)

        # bitwise and operation to extract foreground/person from the frame
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # bitwise and operation to extract the background (mountain) area
        background = cv2.bitwise_and(mountain, mountain, mask=inv_mask)

        # combining the foreground and background to get the final image
        final_image = cv2.add(foreground, background)

        # show it
        cv2.imshow('frame', final_image)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
