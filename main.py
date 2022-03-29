import cv2
import numpy as np
import pytesseract
import os

# setting tesseract path
pytesseract.pytesseract.tesseract_cmd = "D:\\Tesseract-OCR\\tesseract.exe"

per = 25
pixelThreshold = 500

# region of interest
roi = [[(232, 358), (850, 386), 'boxtext', 'clientID'], 
[(498, 504), (820, 526), 'text', 'cmName'], 
[(822, 506), (1142, 524), 'numtext', 'targetActNo'], 
[(628, 716), (1144, 744), 'text', 'custID'], 
[(244, 1028), (542, 1070), 'text', 'Name'], 
[(246, 1078), (542, 1150), 'signtext', 'signature']]

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
    
    # parameters taken from documentation - finding relation
    M, _ = cv2.findHomography(srcPoints,destPoints,cv2.RANSAC,5.0)

    imgScan = cv2.warpPerspective(img,M,(w,h))
    # imgScan = cv2.resize(imgScan, (w//3,h//3))
    # cv2.imshow(y,imgScan)

    # copying and adding mask
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []
    print(f'################ Extracting data from form {j} ################')
    # iterate through roi and mark the regions with "GREEN" colour
    for x,r in enumerate(roi):
        # marking the regions
        cv2.rectangle(imgMask, (r[0][0],r[0][1]), (r[1][0],r[1][1]), (0,255,0), cv2.FILLED)
        # combining the copy and mask to obtain the form with regions marked
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        # cropping the image
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x),imgCrop)

        if r[2] == "text":
            print(f"{r[3]} : {pytesseract.image_to_string(imgCrop)}")
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] == "box":
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgThreshold = cv2.threshold(imgGray,170,255, cv2.THRESH_BINARY_INV)[1]
            # finding dark and light regions
            totalPixels = cv2.countNonZero(imgThreshold)
            if totalPixels > pixelThreshold:
                totalPixels = 1
            else:
                totalPixels = 0
            print(f"{r[3]} : {totalPixels}")
            myData.append(totalPixels)
        # works in case of plain forms and fonts
        # cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
    # writing into csv file
    with open("./dataOutput.csv","a+") as f:
        for data in myData:
            f.write((str(data.replace("\n",""))+","))
        f.write("\n")
    # print(myData)
    imgShow = cv2.resize(imgShow, (w//3,h//3))
    cv2.imshow(y+"2",imgShow)

# cv2.imshow("Output",imgQ)
cv2.waitKey(0)