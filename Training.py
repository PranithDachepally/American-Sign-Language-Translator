import cv2 as cv
import numpy as np
from numpy.linalg import norm

class StatModel(object):
    def load(self,fn):
        self.svm.load(fn)
    def save(self,fn):
        self.svm.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.svm = cv.ml.SVM_create()
        self.svm.setType(cv.ml.SVM_C_SVC)
        self.svm.setKernel(cv.ml.SVM_RBF)
        self.svm.setC(C)
        self.svm.setGamma(gamma)

    def train(self,samples,labels):
        self.svm.train(samples, cv.ml.ROW_SAMPLE, labels)

    def predict(self,samples):
        return self.svm.predict(samples)

def preprocess_hog(trainingData):
    samples = []
    for img in trainingData:
        gx = cv.Sobel(img,cv.CV_32F,1,0)
        gy = cv.Sobel(img,cv.CV_32F,0,1)
        mag, ang = cv.cartToPolar(gx,gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:100,:100],bin[100:,:100],bin[:100,100:],bin[100:,100:]
        mag_cells = mag[:100,:100],mag[100:,:100],mag[:100,100:],mag[100:,100:]
        hists = [np.bincount(b.ravel(),m.ravel(),bin_n) for b,m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        #transform to Hellinger kernel
        eps =1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        samples.append(hist)

    return np.float32(samples)


def hog_simple(img):
    samples = []
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bin[:100, :100], bin[100:, :100], bin[:100, 100:], bin[100:, 100:]
    mag_cells = mag[:100, :100], mag[100:, :100], mag[:100, 100:], mag[100:, 100:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps
    samples.append(hist)

    return np.float32(samples)

def trainSVM():
    trainingData = []
    num = 17
    #trainingData
    for i in range(65,num+65):
        for j in range(1,401):
            print('Class '+chr(i)+' is being loaded')
            trainingData.append(cv.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))
    labels = np.repeat(np.arange(1,num+1),400)
    samples = preprocess_hog(trainingData)
    svm = SVM(C = 2.67, gamma = 5.383)
    svm.train(samples,labels)
    print('--------------------------------------------------------Training is Completed--------------------------------------------------------')
    return svm

def predict(svm,img):
    samples = hog_simple(img)
    resp = svm.predict(samples)
    return resp