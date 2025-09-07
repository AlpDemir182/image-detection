import cv2
import numpy as np

image1 = cv2.imread("Object Detection/Circles4.png", cv2.IMREAD_COLOR )
# the image is in black and white but we are giving the computer an illusion that it is in color

grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# we are converting the image into greyscale as for the requirement for object detection by the computer

blurgrey = cv2.blur(grey, (3,3))

detectedcircles = cv2.HoughCircles(blurgrey, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2= 30, minRadius=1, maxRadius=75)
# houghcircles is a function to find the circles, blurgrey is a grid of the image, houghgradient is used to detect the circles
# number 1 is the resolution, 20 is the minimum distance between the circles, parameters means the machine will detect the circle even if there is a minor issue, the radius define the smallest and largest circles.

if detectedcircles is not None:
    detectedcircles = np.uint16(np.around(detectedcircles))
    #  the circles are converted into numbers using unified integer datatype

    for i in detectedcircles[0,:]:
        x,y,z = i[0], i[1], i[2]
        cv2.circle(image1, (x,y), z, (0,255,0), 5)
        cv2.circle(image1, (x,y), 1, (255, 0, 0), 5)
        cv2.imshow("detectedcircles", image1)
        cv2.waitKey(0)


cv2.imshow("original image", image1)
cv2.waitKey(0)

#cv2.imshow("grey image", grey)
#cv2.waitKey(0)

#cv2.imshow("blurgrey", blurgrey)
#cv2.waitKey(0)

cv2.destroyAllWindows()