from sre_constants import SUCCESS
import cv2
import numpy as np


capture = cv2.VideoCapture("deel2\IMG_4011.mp4") #the video which we search the image on
targetImg = cv2.imread("deel2\edm_image.png") #the image to detect
overLayImg = cv2.imread("deel2\davy.jpg") #the picture we place 


height,width,c = targetImg.shape
overLayImg = cv2.resize(overLayImg, (width, height))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(targetImg,None)


while capture.isOpened():
    status,frame = capture.read()

    kp2, des2 = orb.detectAndCompute(frame,None)


    bfMatcher = cv2.BFMatcher()

    matches = bfMatcher.knnMatch(des1,des2,k=2)
    goodMatches =[]
    for m,n in matches:
        if m.distance <0.75 * n.distance:
            goodMatches.append(m)

    imgFeatures = cv2.drawMatches(targetImg,kp1,frame,kp2,goodMatches,None,flags=2)
    
    if len(goodMatches)> 20:
        srcpts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
        dstpts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)

        matrix, mask = cv2.findHomography(srcpts,dstpts,cv2.RANSAC,5)


        pts = np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)

        imgWarp = cv2.warpPerspective(overLayImg,matrix,(frame.shape[1],frame.shape[0]))

        maskNew = np.zeros((frame.shape[0],frame.shape[1]), np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)], (255,255,255) )
        maskInv = cv2.bitwise_not(maskNew)
        imAug = frame.copy()

        imAug = cv2.bitwise_and(imAug, imAug, mask=maskInv)

        imAug = cv2.bitwise_or(imAug,imgWarp)

        cv2.imshow("features",imAug)
    
    #cv2.imshow("frame",imgFeatures)
    #cv2.imshow("target", targetImg)
    #cv2.imshow("overlay",overLayImg)
    cv2.waitKey(30)

