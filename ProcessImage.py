#File:   ProcessImage.py
#Author: Brendan Waters
#Email:  b101@umbc.edu
#
#Description:
#  This program will take a filepath to an image and classify that image as
# a face, a hat, a hashtag, a heart, or a dollar sign
#
#References
# http://www.scipy-lectures.org/packages/scikit-image/
# http://blog.yhat.com/posts/image-processing-with-scikit-image.html
# http://scikit-learn.org/stable/modules/svm.html
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

import sys
from sklearn import svm, metrics
import skimage
from skimage import io
from skimage import filters
from skimage.color import rgb2gray
import numpy as np
import os

#opens image, converts it to black and white and returns it as nparray
def getImage(filepath):

    image = io.imread(filepath)
    gray_image = rgb2gray(image)
    binary_image = np.where(gray_image > np.mean(gray_image), 1.0, 0.0)

    return binary_image

#creates SVM, using training set
def createSVM():

    #SVM lists
    data = []
    classification = []
    

    #get all the training data
    for i in range(0, 100):
        
        #make the filepaths
        number = ""
        if(i < 10):
            number = "0" + str(i)
        else:
            number = str(i)
        
        facePath = "Training/01/" + number + ".jpg"
        hatPath = "Training/02/" + number + ".jpg"
        hashtagPath = "Training/03/" + number + ".jpg"
        heartPath = "Training/04/" + number + ".jpg"
        dollarSignPath = "Training/05/" + number + ".jpg"
        
        if(os.path.exists(facePath)):
            face = getImage(facePath)
            data.append(face)
            classification.append(1)
        if(os.path.exists(hatPath)):
            hat = getImage(hatPath)
            data.append(hat)
            classification.append(2)
        if(os.path.exists(hashtagPath)):
            hashtag = getImage(hashtagPath)
            data.append(hashtag)
            classification.append(3)
        if(os.path.exists(heartPath)):
            heart = getImage(heartPath)
            data.append(heart)
            classification.append(4)
        if(os.path.exists(dollarSignPath)):
            dollarSign = getImage(dollarSignPath)
            data.append(dollarSign)
            classification.append(5)
            
    #now do SVM stuff
    n_samples = len(data)
    shaped_data = np.asarray(data).reshape((n_samples, -1))
    lin_clf = svm.LinearSVC()
    lin_clf.fit(shaped_data, classification)

    #CODE USED TO TEST ACCURACY
    #half_n_samples = int(n_samples / 2)
    #lin_clf.fit(shaped_data[:half_n_samples], classification[:half_n_samples])
    #expected = classification[half_n_samples:]
    #predicted = lin_clf.predict(shaped_data[half_n_samples:])
    #print("Classification report for classifier %s:\n%s\n" % (lin_clf, metrics.classification_report(expected, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    return lin_clf

def main():

    if(len(sys.argv) != 2):
        print("Usage: python ProcessImage.py filepath")
        exit()

    filepath = sys.argv[1]

    if(not os.path.exists(filepath)):
        print("Invlaid filepath! Please try again")
        exit()

    supportVectorMachine = createSVM()

    image = getImage(filepath)

    image_array = image.reshape(1, -1)

    prediction_number = supportVectorMachine.predict(image_array)

    #change prediction into corr. string
    prediction = "The image is of a "
    if(prediction_number[0] == 1):
        prediction += "face"
    elif(prediction_number[0] == 2):
        prediction += "hat"
    elif(prediction_number[0] == 3):
        prediction += "hashtag"
    elif(prediction_number[0] == 4):
        prediction += "heart"
    elif(prediction_number[0] == 5):
        prediction += "dollar sign"

    print(prediction)

if __name__ == "__main__": main()
