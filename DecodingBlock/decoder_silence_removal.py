#read wav files, get the data

from scipy.io.wavfile import read
import exceptions
import sys
import numpy
from sklearn.svm import SVC
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

eps = 0.00000001
    
def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def stFeatureExtraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.
    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    # signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    # print 'length=',N
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    stFeatures = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude        
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((1))
        curFV[0] = stEnergy(x)                           # short-term energy
        if countFrames == 1:
            stFeatures = curFV                                        # initialize feature matrix (if first frame)
        else:
            stFeatures = numpy.concatenate((stFeatures, curFV))    # update feature matrix
        Xprev = X.copy()
    # print 'countFrames=',countFrames
    return numpy.array(stFeatures)