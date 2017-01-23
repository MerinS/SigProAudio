import encoder_silence_removal as enc_SilRem
import encodewatermark as enc_Wat
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

watermark_original = '1100111000' 
           
U           = 4    #no of frames per unit
B           = 10    #no of units per block
Bits_Block  = 2    #no of bits per block
Tiles_bits  = 30
Sync_bits   = 220
Total_tiles = 280

# no of tiles to watermark 10*28 = 280
# 2*20=40 watermark and 100 sync bits 
Positions          = [0.0 , 1.0 , 2.0 , 3.0 , 4.0 , 5.0 , 6.0 , 7.0 , 8.0 , 9.0 , 10.0 , 11.0 , 12.0 , 13.0 , 14.0 , 15.0 , 16.0 , 17.0 , 18.0 , 19.0 , 20.0 , 21.0 , 22.0 , 23.0 , 24.0 , 25.0 , 26.0 , 27.0 , 28.0 , 29.0 , 30.0 , 31.0 , 32.0 , 33.0 , 34.0 , 35.0 , 36.0 , 37.0 , 38.0 , 39.0 , 40.0 , 41.0 , 42.0 , 43.0 , 44.0 , 45.0 , 46.0 , 47.0 , 48.0 , 49.0 , 50.0 , 51.0 , 52.0 , 53.0 , 54.0 , 55.0 , 56.0 , 57.0 , 58.0 , 59.0 , 60.0 , 61.0 , 62.0 , 63.0 , 64.0 , 65.0 , 66.0 , 67.0 , 68.0 , 69.0 , 70.0 , 71.0 , 72.0 , 73.0 , 74.0 , 75.0 , 76.0 , 77.0 , 78.0 , 79.0 , 80.0 , 81.0 , 82.0 , 83.0 , 84.0 , 85.0 , 86.0 , 87.0 , 88.0 , 89.0 , 90.0 , 91.0 , 92.0 , 93.0 , 94.0 , 95.0 , 96.0 , 97.0 , 98.0 , 99.0 , 100.0 , 101.0 , 102.0 , 103.0 , 104.0 , 105.0 , 106.0 , 107.0 , 108.0 , 109.0 , 110.0 , 111.0 , 112.0 , 113.0 , 114.0 , 115.0 , 116.0 , 117.0 , 118.0 , 119.0 , 120.0 , 121.0 , 122.0 , 123.0 , 124.0 , 125.0 , 126.0 , 127.0 , 128.0 , 129.0 , 130.0 , 131.0 , 132.0 , 133.0 , 134.0 , 135.0 , 136.0 , 137.0 , 138.0 , 139.0 , 140.0 , 141.0 , 142.0 , 143.0 , 144.0 , 145.0 , 146.0 , 147.0 , 148.0 , 149.0 , 150.0 , 151.0 , 152.0 , 153.0 , 154.0 , 155.0 , 156.0 , 157.0 , 158.0 , 159.0 , 160.0 , 161.0 , 162.0 , 163.0 , 164.0 , 165.0 , 166.0 , 167.0 , 168.0 , 169.0 , 170.0 , 171.0 , 172.0 , 173.0 , 174.0 , 175.0 , 176.0 , 177.0 , 178.0 , 179.0 , 180.0 , 181.0 , 182.0 , 183.0 , 184.0 , 185.0 , 186.0 , 187.0 , 188.0 , 189.0 , 190.0 , 191.0 , 192.0 , 193.0 , 194.0 , 195.0 , 196.0 , 197.0 , 198.0 , 199.0 , 200.0 , 201.0 , 202.0 , 203.0 , 204.0 , 205.0 , 206.0 , 207.0 , 208.0 , 209.0 , 210.0 , 211.0 , 212.0 , 213.0 , 214.0 , 215.0 , 216.0 , 217.0 , 218.0 , 219.0 , 220.0 , 221.0 , 222.0 , 223.0 , 224.0 , 225.0 , 226.0 , 227.0 , 228.0 , 229.0 , 230.0 , 231.0 , 232.0 , 233.0 , 234.0 , 235.0 , 236.0 , 237.0 , 238.0 , 239.0 , 240.0 , 241.0 , 242.0 , 243.0 , 244.0 , 245.0 , 246.0 , 247.0 , 248.0 , 249.0 , 250.0 , 251.0 , 252.0 , 253.0 , 254.0 , 255.0 , 256.0 , 257.0 , 258.0 , 259.0 , 260.0 , 261.0 , 262.0 , 263.0 , 264.0 , 265.0 , 266.0 , 267.0 , 268.0 , 269.0 , 270.0 , 271.0 , 272.0 , 273.0 , 274.0 , 275.0 , 276.0 , 277.0 , 278.0 , 279.0]
Positions_srambled = [52.0 , 29.0 , 79.0 , 203.0 , 160.0 , 124.0 , 158.0 , 176.0 , 61.0 , 150.0 , 131.0 , 26.0 , 117.0 , 146.0 , 214.0 , 276.0 , 15.0 , 153.0 , 181.0 , 184.0 , 3.0 , 5.0 , 72.0 , 7.0 , 253.0 , 268.0 , 173.0 , 65.0 , 226.0 , 245.0 , 24.0 , 55.0 , 179.0 , 166.0 , 102.0 , 206.0 , 90.0 , 177.0 , 81.0 , 139.0 , 100.0 , 204.0 , 6.0 , 68.0 , 41.0 , 137.0 , 205.0 , 169.0 , 109.0 , 93.0 , 220.0 , 8.0 , 132.0 , 172.0 , 127.0 , 40.0 , 92.0 , 56.0 , 138.0 , 251.0 , 53.0 , 235.0 , 47.0 , 18.0 , 168.0 , 64.0 , 271.0 , 216.0 , 12.0 , 54.0 , 208.0 , 44.0 , 136.0 , 222.0 , 142.0 , 19.0 , 22.0 , 60.0 , 170.0 , 165.0 , 156.0 , 84.0 , 154.0 , 229.0 , 62.0 , 279.0 , 213.0 , 129.0 , 157.0 , 23.0 , 125.0 , 244.0 , 116.0 , 2.0 , 67.0 , 97.0 , 108.0 , 261.0 , 87.0 , 191.0 , 225.0 , 78.0 , 130.0 , 228.0 , 211.0 , 201.0 , 33.0 , 119.0 , 49.0 , 118.0 , 223.0 , 77.0 , 243.0 , 89.0 , 264.0 , 252.0 , 43.0 , 98.0 , 73.0 , 215.0 , 14.0 , 48.0 , 262.0 , 193.0 , 140.0 , 50.0 , 250.0 , 259.0 , 167.0 , 85.0 , 20.0 , 265.0 , 182.0 , 232.0 , 74.0 , 196.0 , 35.0 , 141.0 , 217.0 , 59.0 , 112.0 , 128.0 , 277.0 , 274.0 , 209.0 , 32.0 , 249.0 , 161.0 , 207.0 , 270.0 , 171.0 , 230.0 , 212.0 , 110.0 , 189.0 , 103.0 , 83.0 , 13.0 , 1.0 , 234.0 , 94.0 , 114.0 , 30.0 , 256.0 , 231.0 , 194.0 , 180.0 , 66.0 , 4.0 , 107.0 , 269.0 , 113.0 , 199.0 , 11.0 , 192.0 , 105.0 , 106.0 , 104.0 , 258.0 , 99.0 , 175.0 , 240.0 , 39.0 , 198.0 , 57.0 , 219.0 , 255.0 , 36.0 , 267.0 , 45.0 , 186.0 , 218.0 , 37.0 , 242.0 , 183.0 , 190.0 , 95.0 , 134.0 , 42.0 , 266.0 , 272.0 , 188.0 , 31.0 , 144.0 , 185.0 , 115.0 , 200.0 , 51.0 , 28.0 , 126.0 , 96.0 , 135.0 , 236.0 , 38.0 , 46.0 , 148.0 , 133.0 , 111.0 , 145.0 , 63.0 , 227.0 , 247.0 , 34.0 , 197.0 , 155.0 , 16.0 , 71.0 , 69.0 , 233.0 , 88.0 , 86.0 , 263.0 , 0.0 , 164.0 , 21.0 , 123.0 , 10.0 , 147.0 , 76.0 , 9.0 , 273.0 , 275.0 , 143.0 , 210.0 , 278.0 , 246.0 , 248.0 , 159.0 , 202.0 , 152.0 , 27.0 , 122.0 , 25.0 , 120.0 , 260.0 , 238.0 , 58.0 , 80.0 , 195.0 , 17.0 , 224.0 , 162.0 , 149.0 , 221.0 , 91.0 , 257.0 , 178.0 , 187.0 , 174.0 , 163.0 , 75.0 , 82.0 , 121.0 , 237.0 , 241.0 , 151.0 , 70.0 , 254.0 , 239.0 , 101.0]
PRN                = [1 , -1 , 1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , 1 , -1 , -1 , -1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , 1 , 1 , -1 , 1 , 1 , 1 , 1 , 1 , -1 , 1 , 1 , -1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , -1 , -1 , -1 , -1 , -1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , 1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , -1 , -1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , -1 , 1 , 1 , 1 , 1 , -1 , -1 , 1 , -1 , 1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , 1 , 1 , -1 , -1 , -1 , -1 , 1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , -1]

# utilities for the LDPC mapping
# The LDPC mapping used is to map an information sequence of 
# 3 bits to a code word of 6 bits. The parity check is as follows-
# Check one - bits 1,2,3,4
# Check one - bits 3,4,6
# Check one - bits 1,4,5
# The code is not super generic
# The multiplier here is obtained by following the row transformations in -->
# https://en.wikipedia.org/wiki/Low-density_parity-check_code
multiplier   = numpy.asarray([[1,0,0,1,0,1],[0,1,0,1,1,1],[0,0,1,1,1,0]])
values_array = numpy.empty(shape=(1,3))


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
        EnergySt = enc_SilRem.stFeatureExtraction(x,Fs,frame_size,step_size)        # extract short-term features
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

        #     plt.subplot(1, 1, 1)
        #     plt.plot(timeX, x)
        #     for s in segmentLimits:
        #         plt.axvline(x=s[0])
        #         plt.axvline(x=s[1])
        #     plt.title('Signal')
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

def LDPC_encode(watermark_original):
    x = len(watermark_original)%3
    if(x!=0):
        if(x==1):
            watermark_original=watermark_original+'00'
        if(x==2):
            watermark_original=watermark_original+'0'

    watermark = str()
    for i in range(len(watermark_original)/3):
        values = watermark_original[(i*3):(i*3)+3]
        for i in range(len(values)):
            values_array[0][i] = int(values[i])
        codeword = numpy.dot(values_array,multiplier)
        
        for i in range(codeword.shape[1]):
            if(codeword[0][i]%2 == 1):
                watermark = watermark+'1'
            else:
                watermark = watermark+'0'
    return watermark


file             = 'input.wav'
rate,data        = enc_Wat.dataread(file)
data             = stereo2mono(data)                        # convert to mono
N                = len(data)
watermarked_data = numpy.empty(N)
watermarked_data = 1*data

# LDPC Encoding
watermark = LDPC_encode(watermark_original)


# once can either choose to do silence detection and removal, in which case
# the resulting watermarked audio will be so much more better in quality
# the last argument is made one if and only if there is a need to remove 
# silence frames from the audio
segment_limits        = silenceRemoval(data,1)
no_blocks             = len(watermark)/Bits_Block
duration_block        = ((B*U)+1)*frame_size/(Fs*(2.0))
duration_block_points = int(duration_block*Fs)
duration_frame        = frame_size/float(Fs)
duration_file         = N/float(Fs)

# TODO - iteration for modifying the leeway involved in watermarking
# the audio in such a way that the segment limits get increased.
total_blocks_watermarked = 0
c  = 0
# U = 4    no of frames per unit
# B = 10   no of units per block
for i in range(len(segment_limits)):
    # figures out the no of blocks within a non silent segment
    no_blocks_segment = int((segment_limits[i][1]-segment_limits[i][0])/duration_block) 
    if(((segment_limits[i][1]-(segment_limits[i][0]+(no_blocks_segment*duration_block)))>duration_frame) or (segment_limits[i][1]+duration_frame<duration_file)):
        if(total_blocks_watermarked+no_blocks_segment<no_blocks):
            for j in range(no_blocks_segment):
                offset  = j*duration_block
                start   = int(floor((segment_limits[i][0]+offset)*Fs))
                end     = start+duration_block_points
                start1  = int(floor((total_blocks_watermarked+j)*Bits_Block))
                end1    = int(floor(((total_blocks_watermarked+j)*Bits_Block)+Bits_Block))
                watermark_expanded = expand_bits(watermark[start1:end1])
                return_data = enc_Wat.watermarking_block(data[start:end],watermark_expanded,Fs,frame_size,step_size)
                for k in range(start,end):
                    watermarked_data[k] = return_data[k-start]
            total_blocks_watermarked = total_blocks_watermarked+no_blocks_segment
        else:
            for j in range(no_blocks-1-total_blocks_watermarked):
                offset  = j*duration_block
                start   = int(floor((segment_limits[i][0]+offset)*Fs))
                end     = start+duration_block_points
                start1  = int(floor((total_blocks_watermarked+j)*Bits_Block))
                end1    = int(floor(((total_blocks_watermarked+j)*Bits_Block)+Bits_Block))
                watermark_expanded = expand_bits(watermark[start1:end1])
                return_data = enc_Wat.watermarking_block(data[start:end],watermark_expanded,Fs,frame_size,step_size)
                for k in range(start,end):
                    watermarked_data[k] = return_data[k-start]
            print 'Watermarking Done'
            c = 1    
            break
if(c==0):
    print 'Insufficient data to watermark bits in'

enc_Wat.datawrite('output.wav',rate,watermarked_data)

# TODO - explore using ICA for watermark decoding
watermarked_data = numpy.array(watermarked_data,dtype=float)

# comparison of the watermarked and the original signal
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(data)
plt.subplot(2, 1, 2)
plt.plot(watermarked_data)
plt.show() 