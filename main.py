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
"""imageGreyScale changes an image to a single color spectrum using user specified weights"""


def imageGreyScale(imageArray, weights):
    # convert image to rgb array with 3 values per index
    # imageArray = numpy.asarray(im)

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
    quantile = int((highest - lowest) / quant)
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
    avg = numpy.average((values * weights))
    return int(avg)


"""averageFilter looks at a pixel and calculates the average of it along with neighboring pixels within a user specified
mask size, accounting for user specified pixel weights. On a new array, this function sets a corresponding pixel to the 
value of that average and iterates until a new, filtered image is created."""


def averageFilter(imArray, maskSize, averageWeights):
    # variables to change array slice size based on mask size
    # maskIndex 1 also used to crop and place pixels correctly in our cropped array
    maskIndex1 = (int((maskSize - 1) / 2))
    maskIndex2 = maskIndex1 + 1
    # we want to crop to deal with edges, here we calculate where it will be
    rowStart = maskIndex1
    colStart = maskIndex1
    colFinish = len(imArray) - maskIndex1
    rowFinish = len(imArray[0]) - maskIndex1

    # create copy of image array size to work on
    filtered = numpy.zeros((len(imArray) - (2 * maskIndex1), len(imArray[0]) - (2 * maskIndex1)))

    # loop through original image
    for i in range(colStart, colFinish):
        for j in range(rowStart, rowFinish):
            # apply the mask
            arraySlice = numpy.asarray(imArray[i - maskIndex1:i + maskIndex2, j - maskIndex1:j + maskIndex2].flatten())
            # assign weighted average of mask to corresponding pixel in image copy
            average = weightedAverage(arraySlice, averageWeights)
            filtered[i - maskIndex1][j - maskIndex1] = average
    return filtered


def convolute(kernel, slice):
    return int(numpy.sum(kernel * slice))


def suppression(magnitudeArray, directionArray):
    filtered = numpy.zeros_like(magnitudeArray)
    rowStart, rowFinish, colStart, colFinish = maskIndices(magnitudeArray, 3)
    for i in range(colStart, colFinish):
        for j in range(rowStart, rowFinish):
            # 0 degree
            if 0 <= directionArray[i][j] < 22.5 or 157.5 < directionArray[i][j] <= 180:
                if magnitudeArray[i][j] >= magnitudeArray[i][j - 1] \
                        and magnitudeArray[i][j] >= magnitudeArray[i][j + 1]:
                    filtered[i][j] = magnitudeArray[i][j]
                else:
                    filtered[i][j] = 0
            # 45 degree
            elif 22.5 <= directionArray[i][j] < 67.5:
                if magnitudeArray[i][j] >= magnitudeArray[i - 1][j + 1] \
                        and magnitudeArray[i][j] >= magnitudeArray[i + 1][j - 1]:
                    filtered[i][j] = magnitudeArray[i][j]
                else:
                    filtered[i][j] = 0
            # 90 degree
            elif 67.5 <= directionArray[i][j] < 112.5:
                if magnitudeArray[i][j] >= magnitudeArray[i + 1][j] \
                        and magnitudeArray[i][j] >= magnitudeArray[i - 1][j]:
                    filtered[i][j] = magnitudeArray[i][j]
                else:
                    filtered[i][j] = 0
            # 135 degree
            else:
                if magnitudeArray[i][j] >= magnitudeArray[i + 1][j + 1] \
                        and magnitudeArray[i][j] >= magnitudeArray[i - 1][j - 1]:
                    filtered[i][j] = magnitudeArray[i][j]
                else:
                    filtered[i][j] = 0
    return filtered


def threshold(imArray):
    high = 20
    low = 10
    weak = 50
    strong = 255
    count = 0
    weakArray = numpy.zeros_like(imArray)

    for i in range(0, len(imArray)):
        for j in range(0, len(imArray[i])):
            if low <= imArray[i][j] < high:
                weakArray[i][j] = 50
                imArray[i][j] = 0
            elif imArray[i][j] >= high:
                imArray[i][j] = 255
            else:
                imArray[i][j] = 0
    return imArray, weakArray


def edgeTracking(imArray, weakArray):
    pixelStack = []
    filtered = numpy.zeros_like(imArray)

    # iterate through strong array and add all strong pixels to stack for examination
    for i in range(1, len(imArray) - 1):
        for j in range(1, len(imArray[0] - 1)):
            if imArray[i][j] == 255:
                filtered[i][j] = 255
                pixelStack.append([i, j])
    # examine neighbors of pixel stack index in weak array
    while len(pixelStack) > 0:
        index1 = pixelStack[0][0]
        index2 = pixelStack[0][1]
        for i in range(index1 - 1, index1 + 1):
            # we must include a break here because python does not have a goto function and creating a separate method
            # for the nested loop would be sloppy
            if len(pixelStack) == 0:
                break
            for j in range(index2 - 1, index2 + 1):
                if weakArray[i][j] > 0:
                    filtered[i][j] = 255
                    weakArray[i][j] = 0
                    # avoid index out of bounds by not appending something with too few neighbors
                    if 0 < i < len(imArray) and 0 < j < len(imArray[0]):
                        pixelStack.append([i, j])
                del pixelStack[0]
                if len(pixelStack) == 0:
                    break
    return filtered


# we dilate with a full kernel.


def simpleDilate(imArray, kernelSize, passes):
    kernelIndex = int((kernelSize - 1) / 2)
    filtered = numpy.zeros_like(imArray)
    for p in range(0, passes):
        for i in range(1, len(imArray) - 1):
            for j in range(1, len(imArray[0] - 1)):
                if imArray[i][j] == 255:
                    for n in range(i - kernelIndex, i + kernelIndex):
                        for m in range(j - kernelIndex, j + kernelIndex):
                            filtered[n][m] = 255
        imArray = numpy.copy(filtered)
    return filtered


def simpleErode(imArray, kernelSize, passes):
    kernelIndex = int((kernelSize - 1) / 2)
    filtered = numpy.copy(imArray)
    for p in range(0, passes):
        for i in range(1, len(imArray) - 1):
            for j in range(1, len(imArray[0] - 1)):
                if imArray[i][j] == 0:
                    for n in range(i - kernelIndex, i + kernelIndex):
                        for m in range(j - kernelIndex, j + kernelIndex):
                            if imArray[n][m] == 255:
                                filtered[n][m] = 0
        imArray = numpy.copy(filtered)
    return filtered


def sobleEdge(imArray, prewittSkip, jahnneSkip):

    hkernel = [1, 0, -1, 2, 0, -2, 1, 0, -1]
    vkernel = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    if prewittSkip == "1":
        hkernel = [1,0,-1,2,0,-2,1,0,-1]
        vkernel = [1,1,1,0,0,0,-1,-1,-1]
    if jahnneSkip == "1":
        hkernel = [3,0,-3,10,0,-10,3,0,-3]
        vkernel = [3,10,3,0,0,0,-3,-10,-3]
    # create copy of image array size to work on
    filtered = numpy.zeros_like(imArray)
    directionArray = numpy.zeros_like(imArray)
    # we use a border to deal with edges, here we calculate where it will be
    rowStart, rowFinish, colStart, colFinish = maskIndices(imArray, 3)

    directionMax = 180
    # loop through original image
    for i in range(colStart, colFinish):
        for j in range(rowStart, rowFinish):
            # apply the mask
            arraySlice = numpy.asarray(imArray[i - 1:i + 2, j - 1:j + 2].flatten())
            # assign weighted average of mask to corresponding pixel in image copy
            hConvolute = convolute(hkernel, arraySlice)
            vConvolute = convolute(vkernel, arraySlice)
            magnitude = numpy.hypot(hConvolute, vConvolute)
            direction = abs(numpy.arctan2(hConvolute, vConvolute) * 180 / numpy.pi)
            # filtered[i][j] = suppression(magnitude, direction, arraySlice)
            filtered[i][j] = magnitude
            directionArray[i][j] = direction
    return filtered, directionArray


def otsu(imArray):
    histogramArray = histogram(imArray)
    revHistogramArray = numpy.flip(histogramArray)
    cumSum = numpy.cumsum(histogramArray)
    """ 
    This section is removed for code readability. 
    
    I have only seen otsu implementations that calculate cumulative sums of partitions of histograms in each iteration. 
    We can instead make one cumulative sum histogram to grab the cumulative sum at an index. To get the cumulative sum 
    after an index, we can reverse an array, create a cumulative sum array of the reversed array, and reverse it. The 
    index at i + 1 of the reversed, summed, then reversed array is the cumulative sum of the elements of the original 
    array after index i. The same process can be done with our intensity array. 
    
    This is unnecessary in this case because otsu is pretty quick, but it would be helpful if doing otsu on a large
    scale. 
    

    Since we are adding to the index, there is a worry when we look at 255 in the cumSum array, as we should look at an
    index that does not exist in the reverse array. However, index 255 of the cumulative sum of an array of size 256
    represents the sum of all of its elements, and the sum of all elements that follow the  all elements is 0, as there 
    simply aren't any. We can append 0 to the end of the reversed cumulative sum without worry. 
    """
    """
    revHistCumSum = numpy.cumsum(revHistogramArray)
    revCumSumSmall = numpy.flip(revHistCumSum)
    # final reverse array size
    revCumSum = numpy.zeros(257)
    # copy reversed array, leaving last index as 0
    for i in range(0, len(revCumSumSmall)):
        revCumSum[i] = revCumSumSmall[i]
"""
    filtered = numpy.zeros_like(imArray)
    pixels = len(imArray) * len(imArray[0])
    pixelWeight = 1.0 / pixels
    currentThresh = -1
    currentValue = -1
    value = -1
    # 256 total buckets, avoid dividing by 0
    intensity = numpy.arange(256)
    for i in range(1, 255):
        sumForward = numpy.sum(histogramArray[i:])
        sumReverse = numpy.sum(histogramArray[:i])

        # this is just... extremely hacky. I don't know how other people avoid dividing by 0 in these cases, I have seen
        # it not addressed at all in other implementations.
        if sumForward == 0:
            sumForward = .01
        if sumReverse == 0:
            sumReverse = .01

        weightForward = sumForward * pixelWeight
        weightReverse = sumReverse * pixelWeight

        bgMean = numpy.sum((intensity[:i] * histogramArray[:i])) / float(sumReverse)
        fgMean = numpy.sum((intensity[i:] * histogramArray[i:])) / float(sumForward)

        value = weightForward * weightReverse * (bgMean - fgMean) ** 2
        if value > currentValue:
            currentThresh = i - 1
            currentValue = value
            print(value)
    for i in range(0, len(imArray)):
        for j in range(0, len(imArray[i])):
            if imArray[i][j] > currentThresh:
                filtered[i][j] = 255
            else:
                filtered[i][j] = 0
    return filtered


# Fast matrix multiplication for euclidian distances between 2 arrays"
def matrixDistance(imArray1, imArray2):
    # sum to the right

    sumSquare1 = numpy.sum(numpy.square(imArray1), axis=1)
    sumSquare2 = numpy.sum(numpy.square(imArray2), axis=1)
    # transpose 2nd array because we pass centroids as 2 sometimes
    multiplied = numpy.dot(imArray1, imArray2.T)

    # add a dimension to first array sum to force 2d output
    distances = numpy.sqrt(abs(sumSquare1[:, numpy.newaxis] + sumSquare2 - 2 * multiplied))
    return distances


# get a random row and col k # o times and pass back an array of coords
def initCentroids(imArray, k):
    row, col = imArray.shape
    randPoints = []
    for i in range(0, k):
        randRow = numpy.random.randint(row)
        randCol = numpy.random.randint(col)
        randPoint = [randRow, randCol]
        randPoints.append(randPoint)
    return numpy.asarray(randPoints)


def assign(imArray, centroids):
    distances = matrixDistance(imArray, centroids)
    row = len(imArray[0])
    # get indices of min values, traverse to right
    clusterIndices = numpy.argmin(distances, axis=1)
    return clusterIndices


def updateCentroids(centroids, clusterIndices, imArray, k):
    newCentroids = numpy.empty(centroids.shape)
    for i in range(0, k):
        newCentroids[i] = numpy.mean(imArray[clusterIndices == i], axis=0)
    return newCentroids


def calcLoss(centroids, clusterIndices, imArray):
    distances = matrixDistance(imArray, centroids)
    loss = float(0)
    for i in range(0, len(imArray[0])):
        loss = loss + numpy.square(distances[i][clusterIndices[i]])
    return loss


def kMeans(imArray, k, iterations, prevLoss=None):
    # give the seed a value for consistent results during testing
    filtered = numpy.empty(imArray)
    numpy.random.seed(7)
    clusters = []
    centroids = initCentroids(imArray, k)
    # tolerance not really set up yet
    tolerance = 1
    for i in range(0, iterations):
        clusterIndices = assign(imArray, centroids)
        centroids = updateCentroids(centroids, clusterIndices, imArray, k)
        loss = calcLoss(centroids, clusterIndices, imArray)
        if i > 0:
            difference = numpy.abs(prevLoss - loss)
            if difference < tolerance:
                break
        prev_loss = loss
        #track best loss
        #decide to break early?


    for i in range(0, clusterIndices):
        pixelValue = int(255/k) * i
        for j in range(0, clusterIndices[0]):
            # assign pixel values to filtered array at cluster indices
            filtered[clusterIndices[0][1]] = pixelValue


"""In this section, we implement the user's settings from the ini file"""

# parse config.ini for user settings
config = configparser.ConfigParser()
config.read('config.ini')

# user set folders for i/o
inputFolder = config["FILES"]['INPUT_FOLDER']
outputFolder = config["FILES"]['OUTPUT_FOLDER']
finalFolder = config["FILES"]['FINAL_FOLDER']

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
linearSize = int(config['FILTERS']['LINEAR_SIZE'])
medianWeights = numpy.asarray(config['FILTERS']['MEDIAN_WEIGHTS'].split(",")).astype(int)

# linear filter params
linearX = numpy.asarray(config['FILTERS']['LINEAR_X'].split(",")).astype(int)
linearY = numpy.asarray(config['FILTERS']['LINEAR_Y'].split(",")).astype(int)

# very ugly, still not used to python or numpy. We want to multiply kernels to create a big 2d one, then flatten
LinearY2D = [[0] * 1 for i in range(linearSize)]
for i in range(0, len(linearY)):
    LinearY2D[i][0] = linearY[i]
linearWeights = (linearX * LinearY2D).flatten()
linearPasses = int(config['FILTERS']['LINEAR_PASSES'])

# Dilation and erosion
dilationSize = int(config['FILTERS']['DILATE_SIZE'])
dilationPasses = int(config['FILTERS']['DILATE_PASSES'])
erosionSize = int(config['FILTERS']['ERODE_SIZE'])
erosionPasses = int(config['FILTERS']['ERODE_PASSES'])

# Choose whether to run certain operations
# choose whether to run part 1 of project
partOne = config['SKIPLIST']['PART_ONE']
greySkip = config['SKIPLIST']['GREY']
saltSkip = config['SKIPLIST']['SALT_PEPPER']
gaussSkip = config['SKIPLIST']['GAUSS_NOISE']
histSkip = config['SKIPLIST']['HIST']
quantSkip = config['SKIPLIST']['QUANT']
histNormSkip = config['SKIPLIST']['HIST_NORM']
linearSkip = config['SKIPLIST']['LINEAR']
medianSkip = config['SKIPLIST']['MEDIAN']
histAvgSkip = config['SKIPLIST']['HIST_AVG']
partOneVerbose = config['SKIPLIST']['ONE_VERBOSE']
partTwo = config['SKIPLIST']['PART_TWO']
sobleSkip = config['SKIPLIST']['SOBLE_SKIP']
cannySkip = config['SKIPLIST']['CANNY_SKIP']
growSkip = config['SKIPLIST']['GROW_SKIP']
shrinkSkip = config['SKIPLIST']['SHRINK_SKIP']
otsuSkip = config['SKIPLIST']['OTSU_SKIP']
kskip = config['SKIPLIST']['K_SKIP']
jahnneSkip = config['SKIPLIST']['JAHNNE_SKIP']
prewittSkip = config['SKIPLIST']['PREWITT_SKIP']
# Set all time tracking variables to 0
greyScaleTotal, spNoiseTotal, gaussianNoiseTotal, histogramTotal, \
histogramNormTotal, quantTotal, medianTotal, linearTotal, msqeTotal = 0, 0, 0, 0, 0, 0, 0, 0, 0

if partOne == "1":
    # Outer loop loops through a list of lists (imageList), contains lists of filenames of image classes
    for i in range(0, len(imageList)):
        # for i in range(0, 1):
        # create a class average variable. We will add each histogram of a class to this, and divide it at loop end
        histogramClassAverage = numpy.zeros(256, int)

        # Inner loop loops through filenames defined by the user in the ini file
        # for fileName in imageList[i]:
        for j in range(0, len(imageList[i])):
            # for j in range(0, 1):
            fileName = imageList[i][j]
            # open image using pillow and begin running methods based on user settings
            im = Image.open(inputFolder + "/" + fileName)
            imArray = numpy.asarray(im)
            if greySkip == "1":
                # call greyscale function for single spectrum conversion
                greyScaleStart = time.time()
                imArray = imageGreyScale(im, greyscaleWeights)
                greyScaleEnd = time.time()
                greyScaleTotal += greyScaleEnd - greyScaleStart
            if saltSkip == "1":
                # call salt and pepper noise function
                spNoiseStart = time.time()
                imArray = spNoise(imArray, saltWeight, pepperWeight)
                spNoiseEnd = time.time()
                spNoiseTotal += spNoiseEnd - spNoiseStart
            if gaussSkip == "1":
                # call gaussian noise function
                gaussianNoiseStart = time.time()
                imArray = gaussianNoise(imArray, mean, var)
                gaussianNoiseEnd = time.time()
                gaussianNoiseTotal += gaussianNoiseEnd - gaussianNoiseStart
            # this is here to appease the IDE
            histogramArray = []
            histogramStartTime = time.time()
            if histSkip == "1":
                # call histogram function, save as array for calculation later
                histogramStartTime = time.time()
                histogramArray = histogram(imArray)

            if histAvgSkip == "1":
                # Add histogram values to class histogram
                histogramClassAverage = numpy.add(histogramClassAverage, histogramArray)
                histogramEndTime = time.time()
                histogramTotal += histogramEndTime - histogramStartTime

            if histNormSkip == "1":
                # call histogram normalization, save new histogram for quantization
                histogramNormStart = time.time()
                imArray, histogramArray = histogramNormalize(imArray, histogramArray)
                histogramNormEnd = time.time()
                histogramNormTotal += histogramNormEnd - histogramNormStart

            if quantSkip == "1":
                # call quantization
                quantStart = time.time()
                imArray, msqe = simpleQuantize(imArray, histogramArray, quant)
                msqeTotal += msqe
                quantEnd = time.time()
                quantTotal += quantEnd - quantStart
                newImage = Image.fromarray(numpy.uint8(imArray))
                newImage.save("greyprep/" + fileName)

            if linearSkip == "1":
                # call linear filter
                linearStart = time.time()
                for n in range(0, linearPasses):
                    imArray = averageFilter(imArray, linearSize, linearWeights)

                linearEnd = time.time()
                linearTotal += linearEnd - linearStart
                newImage = Image.fromarray(numpy.uint8(imArray))
                newImage.save("edgeprep/" + fileName)

            if medianSkip == "1":
                # call median filter
                medianStart = time.time()
                imArray = medianFilter(imArray, medianSize, medianWeights)
                medianEnd = time.time()
                medianTotal += medianEnd - medianStart

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
        plt.close()
    if partOneVerbose == "1":
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

        totalTimePer = (greyScaleTotal + spNoiseTotal + gaussianNoiseTotal + histogramTotal + histogramNormTotal
                        + quantTotal + medianTotal + linearTotal) / totalImages
        print("Total Time Per Image: " + str(totalTimePer))

# part 2
if partTwo == "1":
    count = 1
    for i in range(0, len(imageList)):
        for j in range(0, len(imageList[i])):
            count += 1
            print(count)
            # for j in range(0, 1):
            fileName = imageList[i][j]
            # open image using pillow and begin running methods based on user settings
            im = Image.open(outputFolder + "/" + fileName)
            imageArray = numpy.asarray(im)
            # soble
            if sobleSkip == "1":
                imageArray, directionArray = sobleEdge(imageArray, prewittSkip, jahnneSkip)
                if prewittSkip == "0" and jahnneSkip == "0":
                    newImage = Image.fromarray(numpy.uint8(imageArray))
                    newImage.save("soble/" + fileName)
                if prewittSkip == "1":
                    newImage = Image.fromarray(numpy.uint8(imageArray))
                    newImage.save("prewitt/" + fileName)
                if jahnneSkip == "1":
                    newImage = Image.fromarray(numpy.uint8(imageArray))
                    newImage.save("jahnne/" + fileName)

            # canny
            if cannySkip == "1":
                imageArray = suppression(imageArray, directionArray)
                imageArray, weakArray = threshold(imageArray)
                imageArray = edgeTracking(imageArray, weakArray)
                newImage.save("canny/" + fileName)
            if growSkip == "1":
                imageArray = simpleDilate(imageArray, dilationSize, dilationPasses)
                newImage = Image.fromarray(numpy.uint8(imageArray))
                newImage.save("dilate/" + fileName)
            if shrinkSkip == "1":
                imageArray = simpleErode(imageArray, erosionSize, erosionPasses)
                newImage = Image.fromarray(numpy.uint8(imageArray))
                newImage.save("erode/" + fileName)
            if otsuSkip == "1":
                imageArray = otsu(imageArray)
                newImage = Image.fromarray(numpy.uint8(imageArray))
                newImage.save("otsu/" + fileName)
            if kskip == "1":
                imageArray = kMeans(imageArray, 2, 10)
            # Save our edited image
            newImage = Image.fromarray(numpy.uint8(imageArray))
            newImage.save(finalFolder + "/" + fileName)
