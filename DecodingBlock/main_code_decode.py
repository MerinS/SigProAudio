import decoder_silence_removal as dec_SilRem
import detectwatermark as det_wat
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
Tiles_bits  = 20
Sync_bits   = 100
Total_tiles = 140
# no of tiles to watermark 10*14 = 140
# 2*20=40 watermark and 100 sync bits 
PRN                = [1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1]
Positions          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]
Positions_srambled = [18, 123, 33, 1, 132, 49, 21, 46, 3, 97, 20, 47, 44, 135, 114, 104, 25, 75, 41, 58, 117, 88, 22, 77, 120, 102, 133, 8, 63, 83, 99, 109, 42, 53, 108, 69, 23, 38, 115, 71, 128, 11, 68, 59, 35, 91, 137, 29, 98, 111, 62, 5, 87, 92, 51, 93, 119, 116, 96, 40, 125, 118, 134, 76, 122, 136, 26, 101, 4, 107, 15, 78, 55, 129, 86, 103, 106, 131, 67, 130, 70, 36, 61, 121, 82, 45, 14, 127, 37, 81, 24, 16, 138, 65, 60, 7, 0, 73, 28, 112, 32, 110, 9, 13, 100, 19, 89, 105, 90, 31, 94, 30, 12, 64, 80, 139, 43, 79, 6, 84, 72, 57, 66, 74, 27, 52, 2, 126, 54, 34, 10, 124, 50, 48, 39, 113, 85, 56, 17, 95]

# length= 525200 and countFrames= 1024
# MER
# 4D4552   01001110
# 0000 0
# 0001 1
# 0010 2
# 0011 3
# 0100 4
# 0101 5
# 0110 6
# 0111 7
# 1000 8
# 1001 9
# 1010 A
# 1011 B 
# 1100 C
# 1101 D
# 1110 E
# 1111 F


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
        ShortTermFeatures = dec_SilRem.stFeatureExtraction(x,Fs,frame_size,step_size)        # extract short-term features
        # ShortTermFeatures.shape[0]=34 (no of features extracted)
        # ShortTermFeatures.shape[1] = 1024 (no of sets of data)

        # Step 2: train binary SVM classifier of low vs high energy frames
        EnergySt = ShortTermFeatures[1, :]                  # keep only the energy short-term sequence (2nd feature)
        E = numpy.sort(EnergySt)                            # sort the energy feature values:
        L1 = int(len(E) / 10)                               # number of 20% of the total short-term windows
        T1 = numpy.mean(E[0:L1])                            # compute "lower" 10% energy threshold
        T2 = numpy.mean(E[-L1:-1])                          # compute "higher" 10% energy threshold
        #high energy class
        Class1 = ShortTermFeatures[:, numpy.where(EnergySt <= T1)[0]]         # get all features that correspond to low energy
        #low energy class
        Class2 = ShortTermFeatures[:, numpy.where(EnergySt >= T2)[0]]         # get all features that correspond to high energy
        featuresSS = [Class1.T, Class2.T]   
        # form the binary classification task and ...
        [featuresNormSS, MEANSS, STDSS] = dec_SilRem.normalizeFeatures(featuresSS)   # normalize and ...
        # print  featuresNormSS
        SVM = dec_SilRem.trainSVM(featuresNormSS, 1.0)# train the respective SVM probabilistic model (ONSET vs SILENCE)

        # print "SVM=",SVM
        # Step 3: compute onset probability based on the trained SVM
        ProbOnset = []
        for i in range(ShortTermFeatures.shape[1]):                    # for each frame
            curFV = (ShortTermFeatures[:, i] - MEANSS) / STDSS         # normalize feature vector
            ProbOnset.append(SVM.predict_proba(curFV.reshape(1,-1))[0][1])           # get SVM probability (that it belongs to the ONSET class)
        ProbOnset = numpy.array(ProbOnset)
        ProbOnset = smoothMovingAvg(ProbOnset, smoothWindow / stStep)  # smooth probability

        # Step 4A: detect onset frame indices:
        ProbOnsetSorted = numpy.sort(ProbOnset)                        # find probability Threshold as a weighted average of top 10% and lower 10% of the values
        Nt = ProbOnsetSorted.shape[0] / 10
        T = (numpy.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * numpy.mean(ProbOnsetSorted[-Nt::]))

        MaxIdx = numpy.where(ProbOnset > T)[0]                         # get the indices of the frames that satisfy the thresholding
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

def expand_bits(watermark_bits):
    bits_expand = numpy.empty(Total_tiles)
    for i in range(Bits_Block):
        if(watermark_bits[i]== '0'):
            dummy = -1
        elif(watermark_bits[i]== '1'):
            dummy = 1
        for j in range(Tiles_bits):
            bits_expand[int(Positions_srambled[(i*Tiles_bits)+j])] = float(PRN[(i*Tiles_bits)+j])*float(dummy)
    for i in range(Bits_Block*Tiles_bits,Total_tiles):
        bits_expand[int(Positions_srambled[i])] = PRN[i];
    return bits_expand


file             = 'output.wav'
rate,data        = det_wat.dataread(file)
data             = stereo2mono(data)                        # convert to mono
watermarked_data = numpy.empty(len(data))
watermarked_data = 1*data
# once can either choose to do silence detection and removal, in which case
# the resulting watermarked audio will be so much more better in quality
# the last argument is made one if and only if there is a need to remove 
# silence frames from the audio
segment_limits = silenceRemoval(data,1)
no_blocks      = len(watermark)/Bits_Block
duration_block = B*U*frame_size/(Fs*(2.0))
duration_frame = frame_size/Fs     

print segment_limits
sys.exit()

# TODO - iteration for modifying the leeway involved in watermarking
# the audio in such a way that the segment limits get increased.
total_blocks_watermarked = 0
c  = 0
# U = 4    no of frames per unit
# B = 10   no of units per block
for i in range(len(segment_limits)):
    # figures out the no of blocks within a non silent segment
    no_blocks_segment = int((segment_limits[i][1]-segment_limits[i][0])/duration_block) 
    if(total_blocks_watermarked+no_blocks_segment<no_blocks):
        for j in range(no_blocks_segment-1):
            offset  = j*duration_block
            start   = int(floor((segment_limits[i][0]+offset)*Fs))
            end     = int(floor(((segment_limits[i][0]+offset)+duration_block)*Fs))
            start1  = int(floor((total_blocks_watermarked+j)*Bits_Block))
            end1    = int(floor(((total_blocks_watermarked+j)*Bits_Block)+Bits_Block))
            watermark_expanded = expand_bits(watermark[start1:end1])
            return_data = det_wat.watermarking_block(data[start:end],watermark_expanded,Fs,frame_size,step_size)
            for k in range(start,end):
                watermarked_data[k] = return_data[k-start]
        total_blocks_watermarked = total_blocks_watermarked+no_blocks_segment
    else:
        for j in range(no_blocks-1-total_blocks_watermarked):
            offset  = j*duration_block
            start   = int(floor((segment_limits[i][0]+offset)*Fs))
            end     = int(floor(((segment_limits[i][0]+offset)+duration_block)*Fs))
            start1  = int(floor((total_blocks_watermarked+j)*Bits_Block))
            end1    = int(floor(((total_blocks_watermarked+j)*Bits_Block)+Bits_Block))
            watermark_expanded = expand_bits(watermark[start1:end1])
            return_data = det_wat.watermarking_block(data[start:end],watermark_expanded,Fs,frame_size,step_size)
            for k in range(start,end):
                watermarked_data[k] = return_data[k-start]
        print 'Watermarking Done'
        c = 1    
        break
if(c==0):
    print 'Insufficient data to watermark bits in'

det_wat.datawrite('output.wav',rate,watermarked_data)

# TODO - explore using ICA for this separation
watermarked_data = numpy.array(watermarked_data,dtype=float)

# comparison of the watermarked and the original signal
plt.subplot(2, 1, 1)
plt.plot(data)
plt.subplot(2, 1, 2)
plt.plot(watermarked_data)
plt.show() 