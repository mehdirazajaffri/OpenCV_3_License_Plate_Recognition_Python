# Main.py

import os
import random
from os import listdir
from os.path import isfile, join

import cv2
from PIL import Image, ImageFilter

import DetectChars
import DetectPlates

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


###################################################################################################
def main():
    for i in [f for f in listdir("LicPlateImages") if isfile(join("LicPlateImages", f))]:

        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training

        if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
            print("\nerror: KNN traning was not successful\n")  # show error message
            return  # and exit program
        # end if

        imgOriginalScene = cv2.imread("LicPlateImages/{}".format(i))  # open image

        if imgOriginalScene is None:  # if image was not read successfully
            print("\nerror: image not read from file \n\n")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return  # and exit program
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates

        cv2.imshow("result/imgOriginalScene-{}".format(i), imgOriginalScene)  # show scene image

        if len(listOfPossiblePlates) == 0:  # if no plates were found
            print("\nno license plates were detected\n")  # inform user no plates were found
        else:  # else
            # if we get in here list of possible plates has at leat one plate

            # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
            licPlate = listOfPossiblePlates[0]

            cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:  # if no chars were found in the plate
                print("\nno characters were detected\n\n")  # show message
                return  # and exit program
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate, i)  # draw red rectangle around plate

            # print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
            print("----------------------------------------")

            # writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # write license plate text on the image

            cv2.imshow("imgOriginalScene", imgOriginalScene)  # re-show scene image

            # cv2.imwrite("result/imgOriginalScene-{}".format(i), imgOriginalScene)  # write image out to file

        # end if else

        # cv2.waitKey(0)  # hold windows open until user presses a key

    return

    # end main


def anonymize_face_simple(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate, name):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

    # cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)  # draw 4 red lines
    # cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    # cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    # cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

    x1y1 = [int(p2fRectPoints[1][0] - p2fRectPoints[1][0] * 0.1),
            int(p2fRectPoints[1][1] - p2fRectPoints[1][1] * 0.1)]

    x2y2 = [int(p2fRectPoints[3][0] + p2fRectPoints[3][0] * 0.1),
            int(p2fRectPoints[3][1])]
    # cv2.rectangle(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[3]), (255, 0, 0), -1)
    # cv2.rectangle(imgOriginalScene, tuple(x1y1), tuple(x2y2), (255, 255, 255), -1)

    cv2.imwrite("result/imgOriginalScene-{}".format(name), imgOriginalScene)

    box = (x1y1[0], x1y1[1], x2y2[0], x2y2[1])
    image = Image.open("result/imgOriginalScene-{}".format(name))
    crop_img = image.crop(box)
    blur_image = crop_img.filter(ImageFilter.GaussianBlur(radius=80))
    image.paste(blur_image, box)
    image.save("result/imgOriginalScene-{}".format(name))

    cv2.imwrite("result/plate-{}.jpeg".format(random.randint(1, 100001)), licPlate.imgPlate)

    # startX, startY, endX, endY = p2fRectPoints.astype('int')

    # cv2.GaussianBlur(imgOriginalScene, (21, 21), 0)

    # orignal_image = cv2.imread("LicPlateImages/nabil.jpg")
    #
    # plate = orignal_image[startY:endY, startX:endX]
    # plate = anonymize_face_simple(plate)
    #
    # orignal_image[startY:endY, startX:endX] = plate
    #
    # cv2.imshow("blurred One", orignal_image)

    # plate = anonymize_face_simple(licPlate.imgPlate)
    # output = np.hstack([imgOriginalScene, plate])
    #
    # imgOriginalScene[p2fRectPoints[1]:p2fRectPoints[2], p2fRectPoints[3]:p2fRectPoints[4]] = plate

    # end function

    ###################################################################################################


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


# end function

###################################################################################################
if __name__ == "__main__":
    main()
