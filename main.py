import cv2
import numpy as np
import pytesseract
import os

# setting tesseract path
pytesseract.pytesseract.tesseract_cmd = "D:\\Tesseract-OCR\\tesseract.exe"
per = 25;
# reading the image and resizing it for convenience
imgQ = cv2.imread("./query-page-1.jpg")
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ, (w//2,h//2))

# creating orb detector
orb = cv2.ORB_create(1000)
# key points and descriptors
kp1, des1 = orb.detectAndCompute(imgQ,None)

# displaying keypoints
# imgKp1 = cv2.drawKeypoints(imgQ,kp1,None)
# cv2.imshow("KeyPointsQuery Image",imgKp1)

# User Forms
path = "UserForms1"
myPicList = os.listdir(path)
# print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    # img = cv2.resize(img, (w//2,h//2))
    # cv2.imshow(y,img)
    kp2, des2 = orb.detectAndCompute(img,None)
    # Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    # matches.sort(key= lambda x: x.distance)
    sorted(matches,key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
    # imgMatch = cv2.resize(imgMatch, (w//3,h//3))
    # cv2.imshow(y,imgMatch)

    # finding relation between query image and test image
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    destPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    # parameters taken from documentation
    M, _ = cv2.findHomography(srcPoints,destPoints,cv2.RANSAC,5.0)

    imgScan = cv2.warpPerspective(img,M,(w,h))
    imgScan = cv2.resize(imgScan, (w//3,h//3))
    cv2.imshow(y,imgScan)

cv2.imshow("Output",imgQ)
cv2.waitKey(0)