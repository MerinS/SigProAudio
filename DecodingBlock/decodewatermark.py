from scipy.io.wavfile import read,write
import exceptions
import sys
import numpy
from sklearn.svm import SVC
from scipy.fftpack import fft,ifft
from scipy.fftpack.realtransforms import dct
from math import pow,exp
# from constants import TH
N_SUBBAND = 8
#critical definitions, ie the index of the frequency array that has 
#the frequency closest to the corresponding critical band rate
#in bark
criticaldefn         = [1, 2, 3, 4, 5, 7, 8, 10, 12, 14, 16, 19, 21, 25, 29, 34, 39, 46, 56, 67, 81, 99, 122, 157]

# multiplier for the frames, B frames per unit, implies, 4 elements
C                    = [1,1,-1,-1]
mfccfilterbank_index = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 23, 25, 28, 30, 35, 37, 43, 46, 53, 57, 65, 70, 80]
Num_subbands         = 28
watermark_strength   = 5
filtbank_ind_scramble= [52 , 56 , 22 , 26 , 7 , 11 , 122 , 130 , 88 , 93 , 140 , 148 , 107 , 113 , 182 , 194 , 160 , 170 , 37 , 41 , 57 , 61 , 12 , 16 , 94 , 99 , 82 , 87 , 47 , 51 , 72 , 76 , 100 , 106 , 27 , 31 , 77 , 81 , 32 , 36 , 131 , 139 , 149 , 159 , 171 , 181 , 17 , 21 , 67 , 71 , 114 , 121 , 62 , 66 , 42 , 46]
filtbank_ind         = [7 , 11 , 12 , 16 , 17 , 21 , 22 , 26 , 27 , 31 , 32 , 36 , 37 , 41 , 42 , 46 , 47 , 51 , 52 , 56 , 57 , 61 , 62 , 66 , 67 , 71 , 72 , 76 , 77 , 81 , 82 , 87 , 88 , 93 , 94 , 99 , 100 , 106 , 107 , 113 , 114 , 121 , 122 , 130 , 131 , 139 , 140 , 148 , 149 , 159 , 160 , 170 , 171 , 181 , 182 , 194]

frame_size           = 512
Halfframe_size       = 256

U                    = 4    #no of frames per unit
B                    = 10    #no of units per block
duration_block_point = ((B*U)+1)*frame_size/(2.0)
Bits_Block           = 2    #no of bits per block
Tiles_bits           = 50
Sync_bits            = 180
Total_tiles          = 280
# no of tiles to watermark 10*28 = 280
# 2*20=40 watermark and 100 sync bits 
Positions            = [0.0 , 1.0 , 2.0 , 3.0 , 4.0 , 5.0 , 6.0 , 7.0 , 8.0 , 9.0 , 10.0 , 11.0 , 12.0 , 13.0 , 14.0 , 15.0 , 16.0 , 17.0 , 18.0 , 19.0 , 20.0 , 21.0 , 22.0 , 23.0 , 24.0 , 25.0 , 26.0 , 27.0 , 28.0 , 29.0 , 30.0 , 31.0 , 32.0 , 33.0 , 34.0 , 35.0 , 36.0 , 37.0 , 38.0 , 39.0 , 40.0 , 41.0 , 42.0 , 43.0 , 44.0 , 45.0 , 46.0 , 47.0 , 48.0 , 49.0 , 50.0 , 51.0 , 52.0 , 53.0 , 54.0 , 55.0 , 56.0 , 57.0 , 58.0 , 59.0 , 60.0 , 61.0 , 62.0 , 63.0 , 64.0 , 65.0 , 66.0 , 67.0 , 68.0 , 69.0 , 70.0 , 71.0 , 72.0 , 73.0 , 74.0 , 75.0 , 76.0 , 77.0 , 78.0 , 79.0 , 80.0 , 81.0 , 82.0 , 83.0 , 84.0 , 85.0 , 86.0 , 87.0 , 88.0 , 89.0 , 90.0 , 91.0 , 92.0 , 93.0 , 94.0 , 95.0 , 96.0 , 97.0 , 98.0 , 99.0 , 100.0 , 101.0 , 102.0 , 103.0 , 104.0 , 105.0 , 106.0 , 107.0 , 108.0 , 109.0 , 110.0 , 111.0 , 112.0 , 113.0 , 114.0 , 115.0 , 116.0 , 117.0 , 118.0 , 119.0 , 120.0 , 121.0 , 122.0 , 123.0 , 124.0 , 125.0 , 126.0 , 127.0 , 128.0 , 129.0 , 130.0 , 131.0 , 132.0 , 133.0 , 134.0 , 135.0 , 136.0 , 137.0 , 138.0 , 139.0 , 140.0 , 141.0 , 142.0 , 143.0 , 144.0 , 145.0 , 146.0 , 147.0 , 148.0 , 149.0 , 150.0 , 151.0 , 152.0 , 153.0 , 154.0 , 155.0 , 156.0 , 157.0 , 158.0 , 159.0 , 160.0 , 161.0 , 162.0 , 163.0 , 164.0 , 165.0 , 166.0 , 167.0 , 168.0 , 169.0 , 170.0 , 171.0 , 172.0 , 173.0 , 174.0 , 175.0 , 176.0 , 177.0 , 178.0 , 179.0 , 180.0 , 181.0 , 182.0 , 183.0 , 184.0 , 185.0 , 186.0 , 187.0 , 188.0 , 189.0 , 190.0 , 191.0 , 192.0 , 193.0 , 194.0 , 195.0 , 196.0 , 197.0 , 198.0 , 199.0 , 200.0 , 201.0 , 202.0 , 203.0 , 204.0 , 205.0 , 206.0 , 207.0 , 208.0 , 209.0 , 210.0 , 211.0 , 212.0 , 213.0 , 214.0 , 215.0 , 216.0 , 217.0 , 218.0 , 219.0 , 220.0 , 221.0 , 222.0 , 223.0 , 224.0 , 225.0 , 226.0 , 227.0 , 228.0 , 229.0 , 230.0 , 231.0 , 232.0 , 233.0 , 234.0 , 235.0 , 236.0 , 237.0 , 238.0 , 239.0 , 240.0 , 241.0 , 242.0 , 243.0 , 244.0 , 245.0 , 246.0 , 247.0 , 248.0 , 249.0 , 250.0 , 251.0 , 252.0 , 253.0 , 254.0 , 255.0 , 256.0 , 257.0 , 258.0 , 259.0 , 260.0 , 261.0 , 262.0 , 263.0 , 264.0 , 265.0 , 266.0 , 267.0 , 268.0 , 269.0 , 270.0 , 271.0 , 272.0 , 273.0 , 274.0 , 275.0 , 276.0 , 277.0 , 278.0 , 279.0]
Positions_srambled   = [52.0 , 29.0 , 79.0 , 203.0 , 160.0 , 124.0 , 158.0 , 176.0 , 61.0 , 150.0 , 131.0 , 26.0 , 117.0 , 146.0 , 214.0 , 276.0 , 15.0 , 153.0 , 181.0 , 184.0 , 3.0 , 5.0 , 72.0 , 7.0 , 253.0 , 268.0 , 173.0 , 65.0 , 226.0 , 245.0 , 24.0 , 55.0 , 179.0 , 166.0 , 102.0 , 206.0 , 90.0 , 177.0 , 81.0 , 139.0 , 100.0 , 204.0 , 6.0 , 68.0 , 41.0 , 137.0 , 205.0 , 169.0 , 109.0 , 93.0 , 220.0 , 8.0 , 132.0 , 172.0 , 127.0 , 40.0 , 92.0 , 56.0 , 138.0 , 251.0 , 53.0 , 235.0 , 47.0 , 18.0 , 168.0 , 64.0 , 271.0 , 216.0 , 12.0 , 54.0 , 208.0 , 44.0 , 136.0 , 222.0 , 142.0 , 19.0 , 22.0 , 60.0 , 170.0 , 165.0 , 156.0 , 84.0 , 154.0 , 229.0 , 62.0 , 279.0 , 213.0 , 129.0 , 157.0 , 23.0 , 125.0 , 244.0 , 116.0 , 2.0 , 67.0 , 97.0 , 108.0 , 261.0 , 87.0 , 191.0 , 225.0 , 78.0 , 130.0 , 228.0 , 211.0 , 201.0 , 33.0 , 119.0 , 49.0 , 118.0 , 223.0 , 77.0 , 243.0 , 89.0 , 264.0 , 252.0 , 43.0 , 98.0 , 73.0 , 215.0 , 14.0 , 48.0 , 262.0 , 193.0 , 140.0 , 50.0 , 250.0 , 259.0 , 167.0 , 85.0 , 20.0 , 265.0 , 182.0 , 232.0 , 74.0 , 196.0 , 35.0 , 141.0 , 217.0 , 59.0 , 112.0 , 128.0 , 277.0 , 274.0 , 209.0 , 32.0 , 249.0 , 161.0 , 207.0 , 270.0 , 171.0 , 230.0 , 212.0 , 110.0 , 189.0 , 103.0 , 83.0 , 13.0 , 1.0 , 234.0 , 94.0 , 114.0 , 30.0 , 256.0 , 231.0 , 194.0 , 180.0 , 66.0 , 4.0 , 107.0 , 269.0 , 113.0 , 199.0 , 11.0 , 192.0 , 105.0 , 106.0 , 104.0 , 258.0 , 99.0 , 175.0 , 240.0 , 39.0 , 198.0 , 57.0 , 219.0 , 255.0 , 36.0 , 267.0 , 45.0 , 186.0 , 218.0 , 37.0 , 242.0 , 183.0 , 190.0 , 95.0 , 134.0 , 42.0 , 266.0 , 272.0 , 188.0 , 31.0 , 144.0 , 185.0 , 115.0 , 200.0 , 51.0 , 28.0 , 126.0 , 96.0 , 135.0 , 236.0 , 38.0 , 46.0 , 148.0 , 133.0 , 111.0 , 145.0 , 63.0 , 227.0 , 247.0 , 34.0 , 197.0 , 155.0 , 16.0 , 71.0 , 69.0 , 233.0 , 88.0 , 86.0 , 263.0 , 0.0 , 164.0 , 21.0 , 123.0 , 10.0 , 147.0 , 76.0 , 9.0 , 273.0 , 275.0 , 143.0 , 210.0 , 278.0 , 246.0 , 248.0 , 159.0 , 202.0 , 152.0 , 27.0 , 122.0 , 25.0 , 120.0 , 260.0 , 238.0 , 58.0 , 80.0 , 195.0 , 17.0 , 224.0 , 162.0 , 149.0 , 221.0 , 91.0 , 257.0 , 178.0 , 187.0 , 174.0 , 163.0 , 75.0 , 82.0 , 121.0 , 237.0 , 241.0 , 151.0 , 70.0 , 254.0 , 239.0 , 101.0]
PRN                  = [1 , -1 , 1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , 1 , -1 , -1 , -1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , 1 , 1 , -1 , 1 , 1 , 1 , 1 , 1 , -1 , 1 , 1 , -1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , -1 , -1 , -1 , -1 , -1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , 1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , -1 , -1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , -1 , 1 , 1 , 1 , 1 , -1 , -1 , 1 , -1 , 1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , 1 , 1 , -1 , -1 , -1 , -1 , 1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , 1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , -1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , 1 , 1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , 1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , 1 , 1 , 1 , 1 , -1 , 1 , -1 , -1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , 1 , -1]

Positions_bits       = [52.0 , 29.0 , 79.0 , 203.0 , 160.0 , 124.0 , 158.0 , 176.0 , 61.0 , 150.0 , 131.0 , 26.0 , 117.0 , 146.0 , 214.0 , 276.0 , 15.0 , 153.0 , 181.0 , 184.0 , 3.0 , 5.0 , 72.0 , 7.0 , 253.0 , 268.0 , 173.0 , 65.0 , 226.0 , 245.0 , 24.0 , 55.0 , 179.0 , 166.0 , 102.0 , 206.0 , 90.0 , 177.0 , 81.0 , 139.0 , 100.0 , 204.0 , 6.0 , 68.0 , 41.0 , 137.0 , 205.0 , 169.0 , 109.0 , 93.0 , 220.0 , 8.0 , 132.0 , 172.0 , 127.0 , 40.0 , 92.0 , 56.0 , 138.0 , 251.0 , 53.0 , 235.0 , 47.0 , 18.0 , 168.0 , 64.0 , 271.0 , 216.0 , 12.0 , 54.0 , 208.0 , 44.0 , 136.0 , 222.0 , 142.0 , 19.0 , 22.0 , 60.0 , 170.0 , 165.0 , 156.0 , 84.0 , 154.0 , 229.0 , 62.0 , 279.0 , 213.0 , 129.0 , 157.0 , 23.0 , 125.0 , 244.0 , 116.0 , 2.0 , 67.0 , 97.0 , 108.0 , 261.0 , 87.0 , 191.0]
PN_bits              = [1.0 , -1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , -1.0 , 1.0 , -1.0 , 1.0 , -1.0 , -1.0 , -1.0 , 1.0 , 1.0 , 1.0 , -1.0 , 1.0 , -1.0 , -1.0 , 1.0 , -1.0 , 1.0 , 1.0 , -1.0 , 1.0 , -1.0 , 1.0 , 1.0 , 1.0 , 1.0 , -1.0 , 1.0 , -1.0 , 1.0 , 1.0 , -1.0 , -1.0 , 1.0 , -1.0 , -1.0 , -1.0 , 1.0 , -1.0 , -1.0 , 1.0 , -1.0 , -1.0 , -1.0 , 1.0 , 1.0 , -1.0 , -1.0 , 1.0 , 1.0 , -1.0 , -1.0 , 1.0 , -1.0 , -1.0 , -1.0 , -1.0 , 1.0 , -1.0 , -1.0 , 1.0 , -1.0 , 1.0 , 1.0 , 1.0 , -1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , -1.0 , 1.0 , 1.0 , -1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , -1.0 , -1.0 , -1.0 , -1.0 , -1.0 , -1.0 , 1.0 , 1.0 , -1.0 , 1.0 , -1.0 , 1.0]
#read the wave file
def dataread(filename):
    try:
        data = read(filename)
        return data
    except IOError:
        print("IOError:Wrong file or file path")
        #TODO we will trace back and add codes for the exit code
        sys.exit()

def datawrite(filename,rate,data):
    try:
        write(filename, rate, data)
    except IOError:
        print("IOError:Wrong file or file path")
        #TODO we will trace back and add codes for the exit code
        sys.exit()

# Block to expand the bits obtained from the bit-expand block
def signexpanded(PNseq,N_unit,factor):
    tmp = []
    for i in range(Num_subbands):
        tmp.append(factor*PNseq[N_unit+(B*i)])

    sign = []
    for i in range(HalfWin):
        sign.append(1)

    for i in range(Num_subbands):
        for j in range(2*i,(2*i)+1):
            sign[j]=tmp[i];
    
    return sign


def hann(wave,length):
    for i in range(length):
        wave[i]=0.81649658092*(1-numpy.cos(2*numpy.pi*i/length))*wave[i]
    return wave


def watermark_decode_block(signal,Fs,frame_size,Step):
    frame_size = int(frame_size)
    Step = int(Step)

    # frequency_array - array of the frequencies in the Freq axis
    # threshold_quiet_vals - array of the thresholds in quiet of the values in the Freq axis
    # bark_array_float - converted from frequency to bark scale
    # bark_array - values rounded after being converted from frequency to bark scale
    # criticaldefn - the index of the frequency array that is closest to the centre frequency in each bark band 

    # Signal normalization
    signal = numpy.double(signal)
    return_signal = signal.copy()

    # TODO analyze the implcations of this
    # signal = signal / (2.0 ** 15)
    # DC = signal.mean()
    # MAX = (numpy.abs(signal)).max()
    # signal = (signal - DC) / MAX

    N              = len(signal)     # total number of samples
    curPos         = 0
    countFrames    = 0
    countUnits     = 0
    prev_watermark = numpy.zeros(frame_size)
    # nFFT        = frame_size / 2
    # print 'New Set'
    # print 'len(signal)',len(signal)
    FFT_abs        = []
    mean_values    = []
    count          = 0
    while (1):                        # for each short-term window until the end of signal
        # take current window
        x            = signal[curPos:curPos+frame_size].copy()
        # hann windowing to allow smooth ends and better concatenation
        x1           = hann(x,frame_size)
        # FFT is performed of the time window
        Xabslog      = 20*numpy.log10(abs(fft(x1))) 
        
        FFT_abs.append(Xabslog-numpy.mean(Xabslog))
        # increment the start
        curPos+=Step 
        count+=1
        # print curPos
        if(curPos+frame_size>duration_block_point+frame_size):
            break

    for i in range(count-2):
        FFT_abs[i]=FFT_abs[i]-FFT_abs[i+2]

    value_array = numpy.zeros([Num_subbands,(U*B)]) 

    for k in range(U*B):
        for i in range(Num_subbands):
            for j in range(filtbank_ind_scramble[2*i],filtbank_ind_scramble[(2*i)+1]):
                value_array[i][k]+=FFT_abs[k][j]
            value_array[i][k]= value_array[i][k]/float(filtbank_ind_scramble[(2*i)+1]-filtbank_ind_scramble[2*i])
  
    # hard coded this as for a given PNseq and a given Position block,it is constant
    # Detect the points where the Bits are encoded
    # PN_bits          = numpy.empty(Tiles_bits*Bits_Block)
    # Positions_bits   = numpy.empty(Tiles_bits*Bits_Block)
    # for i in range(Bits_Block):
    #     for j in range(Tiles_bits):
    #         PN_bits[(i*Tiles_bits)+j] = PRN[(i*Tiles_bits)+j]
    #         Positions_bits[(i*Tiles_bits)+j] = int(Positions_srambled[(i*Tiles_bits)+j])
    
    # Detect the points where the sync bits are put in
    # PN_bits          = numpy.empty(Sync_bits)
    # Positions_bits   = numpy.empty(Sync_bits)
    # for i in range(Bits_Block*Tiles_bits,Total_tiles):
    #     PN_bits[(i-Bits_Block*Tiles_bits)] = PRN[i]
    #     Positions_bits[(i-Bits_Block*Tiles_bits)] = int(Positions_srambled[i])

    # print 'count',count
    FrameIndicator = numpy.ones(Num_subbands*U*B)
    # U   = 4    #no of frames per unit
    # B   = 10    #no of units per block 

    Sum_values1 = numpy.zeros(Bits_Block)
    Sum_values2 = numpy.zeros(Bits_Block)
    Sum_values3 = numpy.zeros(Bits_Block)
    for i in range(B):        
        for j in range(Num_subbands):
            if Positions[(i*Num_subbands)+j] in Positions_bits:
                value = Positions_bits.index(Positions[(i*Num_subbands)+j])
                p     = int(value/Tiles_bits)
                # print 'units,subbands',i,j
                # add a few extra elements to avoid the huge ass sums in the FFT
                for k in range(U):     
                        Sum_values1[p]+=(value_array[j][((i*U)+k)]*PN_bits[value])
                        # print PN_bits[value],value_array[j][((i*U)+k)]*PN_bits[value],Sum_values1[p]
                        Sum_values2[p]+=pow(value_array[j][((i*U)+k)],2)
                        Sum_values3[p]+=1
     
    # print Sum_values1,Sum_values2,Sum_values3
    for i in range(len(Sum_values1)):
        if(Sum_values1[i]>0):
            print '1',
        else:
            print '0',
    # sys.exit()
    #                 FrameIndicator[(i*U*Num_subbands)+(k*Num_subbands)+j] = PRN[value]
    #             # sys.exit()

    # print FrameIndicator
    # print len(Positions_bits)


    # def expand_bits(watermark_bits):
    # bits_expand = numpy.empty(Total_tiles)
    # for i in range(Bits_Block):
    #     if(watermark_bits[i]== '0'):
    #         dummy = -1
    #     elif(watermark_bits[i]== '1'):
    #         dummy = 1
    #     for j in range(Tiles_bits):
    #         bits_expand[int(Positions_srambled[(i*Tiles_bits)+j])] = float(PRN[(i*Tiles_bits)+j])*float(dummy)
    # for i in range(Bits_Block*Tiles_bits,Total_tiles):
    #     bits_expand[int(Positions_srambled[i])] = PRN[i];
    # return bits_expand