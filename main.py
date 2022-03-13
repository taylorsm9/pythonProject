# used to parse config.ini
import configparser
import numpy
import time
# used to open images, convert images into arrays, and convert arrays into images. Not used for processing
from PIL import Image
# used for plotting figures and saving them as images
import matplotlib.pyplot as plt
# to calculate weighted median, unfortunately not much faster than a naive implementation

"""imageGreyScale changes an image to a single color spectrum using user specified weights"""


def imageGreyScale(im, weights):
    # convert image to rgb array with 3 values per index
    imageArray = numpy.asarray(im)

    # make single spectrum float array with 1 value per index from dot product of rgb array and weights
    greyScale = numpy.dot(imageArray, weights)

    # round float array to int
    greyScale = greyScale.astype(int)

    # create 8 bit single spectrum image using pillow
    return greyScale


"""spNoise simultaniously salts and peppers an image using user specified weights. See config file for weight info"""


def spNoise(imageArray, saltWeight, pepperWeight):
    # decide how many pixels to salt or pepper based on number of pixels in image
    arrayElements = len(imageArray) * len(imageArray[0])
    saltPixels = int(arrayElements * (saltWeight + pepperWeight))

    # decide which specific pixel to salt or pepper by randomly getting row and col index values
    rowLength = len(imageArray[0])
    for i in range(0, saltPixels - 1):
        randomPixel = numpy.random.randint(arrayElements)
        index1Value = int(randomPixel / rowLength)
        index2Value = randomPixel % rowLength

        # decide whether to salt or pepper the chosen pixel
        randomBetweenSP = numpy.random.uniform(0, saltWeight + pepperWeight)
        if randomBetweenSP - saltWeight > 0:
            saltOrPepper = 0
        else:
            saltOrPepper = 255

        # change value of the randomly selected pixel to 0 or 255 based on prev operation
        imageArray[int(randomPixel / rowLength)][randomPixel % rowLength] = saltOrPepper
    return imageArray


"""gaussianNoise Creates an array of normally distributed pixel values with user specified mean and varience, adds them 
to our image. 

Some data loss is possible here, given that the values of an index meant to represent a pixel could be above 255 or 
below 0. We could try to quantize the image to 256 values, but we would still experience data loss by using the very 
simple quantization method we implemented for this project. Instead, the values are simply clipped to fit"""


def gaussianNoise(imArray, mean, var):
    sigma = var ** .5

    # create a new array, matching imArray, of gaussian noise with user specified mean and var
    gauss = numpy.random.normal(mean, sigma, (len(imArray), len(imArray[0]))).astype(int)

    # add the gaussian noise to the original array and cut off values out of range
    imArray += gauss
    imArray = numpy.clip(imArray, 0, 255)

    return imArray


"""Histogram method creates an array with 256 indices corresponding to pixel values and counts the occurances of each
value in a given image. A section can be uncommented to save individual histograms"""


def histogram(imArray):
    # initialize array at 0 with values for single spectrum 256
    histogramArray = numpy.zeros(256, int)

    # for each pixel in the image, add its value to the frequency array
    for i in imArray:
        for j in i:
            histogramArray[j] += 1

    # return the array for use for normalization
    """
    This section can be uncommented to save individual histograms, not required and probably not desirable
    plt.plot(histogramArray)
    plt.title(fileName + " Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Occurrences")
    plt.savefig(outputFolder + "/" + fileName[0:len(fileName) - 4:1] + "Histogram.png")
    plt.close() """
    return histogramArray


def histogramNormalize(imageArray, histogram):
    cumSum = numpy.cumsum(histogram)
    # evaluates to 255 or less
    normal = cumSum.max() - cumSum.min()
    # remove lowest value and multiply by remaining # of available pixel values
    n = (cumSum - cumSum.min()) * 255

    # dividing these can only give us values under 256
    normalizedCumSum = (n / normal).astype(int)

    # make 2d array 1d
    flattened = imageArray.flatten()

    # assign each pixel value in our flattened image to its index in normalizedCumSum
    imageNormalized = normalizedCumSum[flattened]

    # split up the 1d array into a 2d array with rows and cols = our original image array
    imageNormalized = numpy.reshape(a=imageNormalized, newshape=imageArray.shape)

    # return our histogram normalized image and our normalized histogram to pass to our quantize method later
    return imageNormalized, normalizedCumSum


"""finds the lowest index of an array with value !=0 by iterating through a histogram"""


def findLowestPopulated(histogram):
    lowest = 0

    for i in range(0, len(histogram) - 1):
        if histogram[i] != 0:
            return histogram[i]
    return lowest


"""finds the highest index of an array with value !=0 by iterating backwards through a histogram"""


def findHighestPopulated(histogram):
    highest = 0

    for i in range(len(histogram) - 1, -1, -1):
        if histogram[i] != 0:
            return i
    return highest


"""simpleQuantize is a fast and lossy quantization method that finds an ideal distance between pixels that are 
distributed across an image evenly (not the case for our images), and then does floor division on pixels to leave at 
most the user specified number of different pixel values. While this is technically quantization, other methods are more
useful for object recognition, which exceeds the scope of project part 1. """


def simpleQuantize(imArray, histogram, quant):
    # find lowest and highest pixel values in image
    lowest = findLowestPopulated(histogram)
    highest = findHighestPopulated(histogram)
    # find range of pixel values and divide by user specified desired # of pixel values
    quantile = (highest - lowest) / quant
    # find the number which you can multiply the quantile without exceeding 256 to increase the distance between values
    quant256 = int(256 / quantile)

    # duplicate array and work on copy for msqe calculation
    quantArray = numpy.copy(imArray)

    # do lossy floor division to reduce # of pixel values, multiply result to increase distance for uint8 array
    quantArray = numpy.uint8(numpy.floor_divide(imArray, quantile))
    quantArray = numpy.uint8((quantArray * quant256))

    # calculate msqe
    differenceArray = numpy.subtract(imArray, quantArray)
    squaredDifference = numpy.square(differenceArray)
    meanSquaredError = squaredDifference.mean()

    return quantArray, meanSquaredError


"""maskIndices takes in an image and a mask size and finds the ideal indices to create an even black border around the
image, allowing us to easily apply a mask"""


def maskIndices(imageArray, maskSize):
    # find remainders (black border size)
    col = len(imageArray)
    row = len(imageArray[0])
    colRemainder = col % maskSize
    rowRemainder = row % maskSize

    # make the border around the image even (rather than losing too many pixels on any one side)
    rowStart = int(numpy.floor(rowRemainder / 2)) + int((maskSize - 1) / 2)
    rowFinish = int(row - (numpy.ceil(rowRemainder / 2))) - int((maskSize - 1) / 2)
    colStart = int(numpy.floor(colRemainder / 2)) + int((maskSize - 1) / 2)
    colFinish = int(col - (numpy.ceil(colRemainder / 2))) - int((maskSize - 1) / 2)
    return rowStart, rowFinish, colStart, colFinish


"""
Naive implementation of weighted median. Not much slower than imported method.
"""
def sliceMedian(medianWeights, arraySlice):
    medianList = []
    for i in range(0, len(arraySlice)):
        for j in range(0, medianWeights[i]):
            medianList.append(arraySlice[i])
    return int(numpy.median(medianList))


"""medianFilter looks at a pixel and calculates the median of it along with neighboring pixels within a user specified
mask size, accounting for user specified pixel weights. On a new array, this function sets a corresponding pixel to the 
value of that median and iterates until a new, filtered image is created."""


def medianFilter(imArray, maskSize, medianWeights):
    # create copy of image array size to work on
    filtered = numpy.zeros_like(imArray)

    # we use a border to deal with edges, here we calculate where it will be
    rowStart, rowFinish, colStart, colFinish = maskIndices(imArray, maskSize)

    # loop through original image
    for i in range(colStart, colFinish):
        for j in range(rowStart, rowFinish):
            # apply the mask
            arraySlice = numpy.asarray([imArray[i - 1:i + 2, j - 1:j + 2].flatten()])
            # assign weighted median of mask to corresponding pixel in image copy
            median = sliceMedian(medianWeights, arraySlice)
            filtered[i][j] = median
    return filtered


"""Simple weighted average takes the average of 2 multiplied arrays"""


def weightedAverage(values, weights):
    return int(numpy.average(values * weights))


"""averageFilter looks at a pixel and calculates the average of it along with neighboring pixels within a user specified
mask size, accounting for user specified pixel weights. On a new array, this function sets a corresponding pixel to the 
value of that average and iterates until a new, filtered image is created."""


def averageFilter(imArray, maskSize, averageWeights):
    # create copy of image array size to work on
    filtered = numpy.zeros_like(imArray)

    # we use a border to deal with edges, here we calculate where it will be
    rowStart, rowFinish, colStart, colFinish = maskIndices(imArray, maskSize)

    # loop through original image
    for i in range(colStart, colFinish):
        for j in range(rowStart, rowFinish):
            # apply the mask
            arraySlice = numpy.asarray([imArray[i - 1:i + 2, j - 1:j + 2].flatten()])
            # assign weighted average of mask to corresponding pixel in image copy
            average = weightedAverage(arraySlice, averageWeights)
            filtered[i][j] = average
    return filtered


"""In this section, we implement the user's settings from the ini file"""

# parse config.ini for user settings
config = configparser.ConfigParser()
config.read('config.ini')

# user set folders for i/o
inputFolder = config["FILES"]['INPUT_FOLDER']
outputFolder = config["FILES"]['OUTPUT_FOLDER']

# filenames for i/o, saved as a list of lists for looping
imageList = []
cylList = config["FILES"]['CYL_LIST'].split(",")
imageList.append(cylList)
interList = config["FILES"]['INTER_LIST'].split(",")
imageList.append(interList)
letList = config["FILES"]['LET_LIST'].split(",")
imageList.append(letList)
modList = config["FILES"]['MOD_LIST'].split(",")
imageList.append(modList)
paraList = config["FILES"]['PARA_LIST'].split(",")
imageList.append(paraList)
superList = config["FILES"]['SUPER_LIST'].split(",")
imageList.append(superList)
svarList = config["FILES"]['SVAR_LIST'].split(",")
imageList.append(svarList)

# import weights for single spectrum conversion and format as float array
greyscaleWeights = numpy.asarray(config['SETTINGS']['GREYSCALE_WEIGHTS'].split(",")).astype(float)

# used for salt and pepper noise
saltWeight = float(config["SETTINGS"]['SALT_WEIGHT'])
pepperWeight = float(config['SETTINGS']['PEPPER_WEIGHT'])

# used for gaussian noise
mean = float(config['SETTINGS']['GAUSS_MEAN'])
var = float(config['SETTINGS']['GAUSS_VAR'])

# number of desired pixel values after quantization
quant = float(config['SETTINGS']['QUANT_TARGET'])

# take in filter mask size and user specified weights
medianSize = numpy.asarray(config['FILTERS']['MEDIAN_SIZE'].split(",")).astype(int)
linearSize = numpy.asarray(config['FILTERS']['LINEAR_SIZE'].split(",")).astype(int)
medianWeights = numpy.asarray(config['FILTERS']['MEDIAN_WEIGHTS'].split(",")).astype(int)
linearWeights = numpy.asarray(config['FILTERS']['LINEAR_WEIGHTS'].split(",")).astype(int)


# Set all time tracking variables to 0
greyScaleTotal, spNoiseTotal, gaussianNoiseTotal, histogramTotal, \
histogramNormTotal, quantTotal, medianTotal, linearTotal, msqeTotal = 0, 0, 0, 0, 0, 0, 0, 0, 0

# Outer loop loops through a list of lists (imageList), contains lists of filenames of image classes
for i in range(0, len(imageList)):
#for i in range(0, 1):

    # create a class average variable. We will add each histogram of a class to this, and divide it at loop end
    histogramClassAverage = numpy.zeros(256, int)

    # Inner loop loops through filenames defined by the user in the ini file
    #for fileName in imageList[i]:
    for j in range(0, len(imageList[i])):
    #for j in range(0, 1):
        fileName = imageList[i][j]
        # open image using pillow and begin running methods based on user settings
        im = Image.open(inputFolder + "/" + fileName)

        # call greyscale function for single spectrum conversion
        greyScaleStart = time.time()
        imArray = imageGreyScale(im, greyscaleWeights)
        greyScaleEnd = time.time()
        greyScaleTotal += greyScaleEnd - greyScaleStart

        # call salt and pepper noise function
        spNoiseStart = time.time()
        imArray = spNoise(imArray, saltWeight, pepperWeight)
        spNoiseEnd = time.time()
        spNoiseTotal += spNoiseEnd - spNoiseStart

        # call gaussian noise function
        gaussianNoiseStart = time.time()
        imArray = gaussianNoise(imArray, mean, var)
        gaussianNoiseEnd = time.time()
        gaussianNoiseTotal += gaussianNoiseEnd - gaussianNoiseStart

        # call histogram function, save as array for calculation later
        histogramStartTime = time.time()
        histogramArray = histogram(imArray)

        # Add histogram values to class histogram
        histogramClassAverage = numpy.add(histogramClassAverage, histogramArray)
        histogramEndTime = time.time()
        histogramTotal += histogramEndTime - histogramStartTime

        # call histogram normalization, save new histogram for quantization
        histogramNormStart = time.time()
        imArray, histogramArray = histogramNormalize(imArray, histogramArray)
        histogramNormEnd = time.time()
        histogramNormTotal += histogramNormEnd - histogramNormStart

        # call cuantization
        quantStart = time.time()
        imArray, msqe = simpleQuantize(imArray, histogramArray, quant)
        msqeTotal += msqe
        quantEnd = time.time()
        quantTotal += quantEnd - quantStart

        # call median filter
        medianStart = time.time()
        imArray = medianFilter(imArray, medianSize, medianWeights)
        medianEnd = time.time()
        medianTotal += medianEnd - medianStart

        # call linear filter
        linearStart = time.time()
        imArray = averageFilter(imArray, linearSize, linearWeights)
        linearEnd = time.time()
        linearTotal += linearEnd - linearStart

        # Save our edited image
        newImage = Image.fromarray(numpy.uint8(imArray))
        newImage.save(outputFolder + "/" + fileName)
    classImages = len(imageList[i][0])
    histogramClassAverage = histogramClassAverage / classImages
    plt.plot(histogramClassAverage)

    plt.title(imageList[i][0][0:3:1] + " Class Average Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Occurrences")
    plt.savefig(outputFolder + "/" + "AverageHistogram_" + imageList[i][0][0:3:1] + ".png")
    plt.show()
    plt.close()

# Print total times per image for each procedure
print("Total Times(Seconds):")
print("Single Spectrum Conversion Total Time: " + str(greyScaleTotal))
print("Salt and Pepper Noise Total Time: " + str(spNoiseTotal))
print("Gaussian Noise Total Time: " + str(gaussianNoiseTotal))
print("Histogram Total Time: " + str(histogramTotal))
print("Histogram Normalization Total Time: " + str(histogramNormTotal))
print("Quantization Total Time: " + str(quantTotal))
print("Median Filter Total Time: " + str(medianTotal))
print("Linear Filter Total Time: " + str(linearTotal))
print()

# count total number of images
totalImages = 0
for i in imageList:
    totalImages += len(i)

# Print average times per image for each procedure
print("Average Times Per Image(Seconds):")
print("Single Spectrum Conversion Average Time: " + str(greyScaleTotal / totalImages))
print("Salt and Pepper Noise Average Time: " + str(spNoiseTotal / totalImages))
print("Gaussian Noise Average Time: " + str(gaussianNoiseTotal / totalImages))
print("Histogram Average Time: " + str(histogramTotal / totalImages))
print("Histogram Normalization Average Time: " + str(histogramNormTotal / totalImages))
print("Quantization Average Time: " + str(quantTotal / totalImages))
print("Median Filter Average Time: " + str(medianTotal / totalImages))
print("Linear Filter Average Time: " + str(linearTotal / totalImages))
print()

# MSQE average per image
print("Mean Squared Error Average Per Image(Total Across All Pixels): ")
print(str(msqeTotal / totalImages))

totalTimePer = (greyScaleTotal + spNoiseTotal + gaussianNoiseTotal + histogramTotal + histogramNormTotal \
+ quantTotal + medianTotal + linearTotal) / totalImages
print("Total Time Per Image: " + str(totalTimePer))
