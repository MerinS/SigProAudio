import decoder_silence_removal as dec_SilRem
import decodewatermark as dec_wat
import LDPC_decoding as LDPC_d
import numpy
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import sys
from math import floor
from sklearn.decomposition import FastICA, PCA

# constants
Fs          = 44100.0
frame_size  = 512
step_size   = 256

watermark   = '01001110110111001101011101010101011110010101010101010101010101010101'
U           = 4    #no of frames per unit
B           = 10   #no of units per block
Bits_Block  = 2    #no of bits per block

def stereo2mono(x):
	'''
	This function converts the input signal (stored in a numpy array) to MONO (if it is STEREO)
	'''
	if x.ndim==1:
		return x
	else:
		if x.ndim==2:
			return ( (x[:,1] / 2) + (x[:,0] / 2) )
		else:
			return -1

""" General utility functions """
def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = numpy.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1], inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
    w = numpy.ones(windowLen, 'd')
    y = numpy.convolve(w/w.sum(), s, mode='same')
    # results are like that of a moving windowed average
    return y[windowLen:-windowLen+1]

def silenceRemoval(x,smoothWindow=0.5, Weight=0.5, plot=True , silenceRemoval_turnon = 1):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - Fs:               sampling freq
         - stWin, stStep:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - Weight:           (optinal) weight factor (0 < Weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - segmentLimits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                    the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''
    #TODO- need to be further analyzed
    stWin = frame_size/Fs
    stStep= step_size/Fs

    if Weight >= 1:
        Weight = 0.99
    if Weight <= 0:
        Weight = 0.01
    
    # Step 1: feature extraction
    #this array has the features all arranged in such a way that one data set is one column with each
    #row having one feature each
    #the features are --> zero crossing rate , short-term energy , short-term entropy of energy , spectral centroid and spread
    #spectral entropy,spectral flux,spectral rolloff,stMFCC,chromaF,chromaF.std
    if(silenceRemoval_turnon == 1):
        EnergySt = dec_SilRem.stFeatureExtraction(x,Fs,frame_size,step_size)        
        # extract short-term features
        E = numpy.sort(EnergySt)                            # sort the energy feature values:
        L1 = int(len(E) / 2)                               # number of 50% of the total short-term windows
        MaxIdx = (numpy.where(EnergySt >= E[-L1])[0])                         # get the indices of the frames that satisfy the thresholding
        i = 0
        timeClusters = []
        segmentLimits = []

        # Step 4B: group frame indices to onset segments
        while i < len(MaxIdx):                                         # for each of the detected onset indices
            curCluster = [MaxIdx[i]]
            if i == len(MaxIdx)-1:
                break
            while MaxIdx[i+1] - curCluster[-1] <= 2:
                curCluster.append(MaxIdx[i+1])
                i += 1
                if i == len(MaxIdx)-1:
                    break
            i += 1
            timeClusters.append(curCluster)
            segmentLimits.append([curCluster[0] * stStep, curCluster[-1] * stStep])

        # Step 5: Post process: remove very small segments:
        minDuration = 0.2
        segmentLimits2 = []
        for s in segmentLimits:
            if s[1] - s[0] > minDuration:
                segmentLimits2.append(s)
        segmentLimits = segmentLimits2

        # uncomment this section only if there is a need to see the plots
        # plotting_start
        # if plot:
        #     timeX = numpy.arange(0, x.shape[0] / float(Fs), 1.0 / Fs)

        #     plt.subplot(2, 1, 1)
        #     plt.plot(timeX, x)
        #     for s in segmentLimits:
        #         plt.axvline(x=s[0])
        #         plt.axvline(x=s[1])
        #     plt.subplot(2, 1, 2)
        #     plt.plot(numpy.arange(0, ProbOnset.shape[0] * stStep, stStep), ProbOnset)
        #     plt.title('Signal')
        #     for s in segmentLimits:
        #         plt.axvline(x=s[0])
        #         plt.axvline(x=s[1])
        #     plt.title('SVM Probability')
        #     plt.show()
        # plotting_end
        return segmentLimits
    else:
        return x

file             = 'output.wav'
rate,data        = dec_wat.dataread(file)
data             = stereo2mono(data)  # convert to mono
N                = len(data)
# once can either choose to do silence detection and removal, in which case
# the resulting watermarked audio will be so much more better in quality
# the last argument is made one if and only if there is a need to remove 
# silence frames from the audio
segment_limits        = silenceRemoval(data,1)
duration_block        = ((B*U)+1)*frame_size/(Fs*(2.0))
duration_block_points = int(duration_block*Fs)
duration_frame        = frame_size/float(Fs)
duration_file         = N/float(Fs)

total_blocks_decoded = 0
c  = 0
# U = 4    no of frames per unit
# B = 10   no of units per block
watermarkdecoded     = []
for i in range(len(segment_limits)):
    # figures out the no of blocks within a non silent segment
    no_blocks_segment = int((segment_limits[i][1]-segment_limits[i][0])/duration_block) 
    if(((segment_limits[i][1]-(segment_limits[i][0]+(no_blocks_segment*duration_block)))>duration_frame) or ((segment_limits[i][1]+(2*duration_frame))<duration_file)):
        for j in range(no_blocks_segment):
            offset        = j*duration_block
            start         = int(floor((segment_limits[i][0]+offset)*Fs))
            end           = int(floor(((segment_limits[i][0]+offset)*Fs)+duration_block_points+(2*frame_size)))   
            # print start,end
            bits_returned = dec_wat.watermark_decode_block(data[start:end],Fs,frame_size,step_size)            
            for k in range(bits_returned):
                watermarkdecoded.append(bits_returned[k])
        total_blocks_decoded = total_blocks_decoded+no_blocks_segment
decoded_sequence = LDPC_d.LDPC_decode(total_blocks_decoded)
print decoded_sequence
print 'DecoDone'