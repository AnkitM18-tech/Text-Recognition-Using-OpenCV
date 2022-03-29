# RegionSelector.py
"""
# Once a region is selected the user is asked to enter the type
and the name:
Type can be 'Text' or 'CheckBox'
Name can be anything
"""

import cv2
import random

path = './'
imageFilename = 'query-page-1.jpg'

scale = 0.5
circles = []
counter = 0
counter2 = 0
point1 = []
point2 = []
myPoints = []
myColour = []


def mousePoints(event, x, y, flags, params):
    global counter, point1, point2, counter2, circle, myColour
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            point1 = int(x // scale), int(y // scale)
            counter += 1
            myColour = (random.randint(0, 2) * 200, random.randint(0, 2) * 200, random.randint(0, 2) * 200)
        elif counter == 1:
            point2 = int(x // scale), int(y // scale)

            type = input('Enter type : ')
            name = input('Enter name : ')
            myPoints.append([point1, point2, type, name])
            counter = 0
        circles.append([x, y, myColour])
        counter2 += 1


image = cv2.imread(path+"/"+imageFilename)
image = cv2.resize(image, (0, 0), None, scale, scale)

while True:
    # To display points
    for x, y, colour in circles:
        cv2.circle(image, (x, y), 3, colour, cv2.FILLED)
    cv2.imshow("Original image", image)
    cv2.setMouseCallback("Original image", mousePoints)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(myPoints)
        break