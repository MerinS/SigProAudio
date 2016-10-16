from scipy.io.wavfile import read
import exceptions
import sys
import numpy
from sklearn.svm import SVC
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from math import pow,exp
# from constants import TH
N_SUBBAND = 8

eps = 0.00000001
bark_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
critical_freq = [50,150,250,350,450,570,700,840,1000,1170,1370,1600,1850,2150,2500,2900,3400,4000,4800,5800,7000,8500,10500,13500]
STRUCTURE = [2,0,1,2,1,2,0,1,0,2,0,0,1,0,1,2,0,1,2,1,1,0,2,1]
PRN = [-1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , 1 , -1 , 1 , 1 , -1 , 1 , -1 , -1 , 1 , -1 , -1 , -1 , -1 , 1 , -1 , -1]
#read the wave file
def dataread(filename):
    try:
        data = read(filename)
        return data
    except IOError:
        print("IOError:Wrong file or file path")
        #LATER we will trace back and add codes for the exit code
        sys.exit()

def bark(f):    
    "Convert a frequency from Hertz to Bark"
    return 13.0 * numpy.arctan(0.76 * f / 1000.0) + 3.5 * numpy.arctan((f / 7500.0) ** 2)

def closeto_critical(f):
    num = 0
    critical_close = []
    for i in range(len(f)):
        if(f[i]>critical_freq[num]):
            if(i-1>=0):
                if(numpy.mean([int(f[i]),int(f[i-1])])>critical_freq[num]):
                    x = (i-1)
                else:
                    x = (i)
            else:
                x = i
            critical_close.append(x)
            num+=1
            if(num==len(critical_freq)):
                break
    return critical_close


def hann(wave,length):
    for i in range(length):
        wave[i]=0.81649658092*(1-numpy.cos(2*numpy.pi*i/length))*wave[i]
    return wave

def SPL_normalise(wave_DB,length):
    maximum_value = max(wave_DB)
    for i in range(length):
        wave_DB[i]=96-maximum_value+wave_DB[i]
        # wave_DB[i] = wave_DB[i]
    return wave_DB

def threshold_quiet(array_frequency,length):
    quiet_threshold = numpy.empty(length)
    quiet_threshold[0] = 0.00944567963877
    for i in range(1,length):
        quiet_threshold[i] = (3.64*pow((array_frequency[i]/1000.0),-0.8))-(6.5*exp(-0.6*pow(((array_frequency[i]/1000.0)-3.3),2)))+(0.001*pow((array_frequency[i]/1000.0),4))
    return quiet_threshold

def tonal_markers_sound(wave_DB,length,barkarray,float_barkarray,criticaldefn,quiet_threshold):
    # print wave_DB
    # if (length>252):
    #     length = 252

    val  = numpy.empty(length)
    Pval = numpy.empty(length)

    for i in [0,1,length-2,length-1]:
        Pval[i] = 0
        val[i] = 0

    # Figures if a point is a tonal maxima, allots a value of 1 to and 0 to non tonal 
    for i in range(2,length-2):
        c = 0
        for j in range(1,3):
            if(wave_DB[i]-wave_DB[i+j]<7 or wave_DB[i]-wave_DB[i-j]<7):
                c = 1
        if c==0 :
            val[i] = 1
            Pval[i]   = 10*numpy.log10(pow(10,(wave_DB[i-1]/10.0))+pow(10,(wave_DB[i]/10.0))+pow(10,(wave_DB[i+1]/10.0)))
        else :
            val[i] = 0

    # Selects only the non tonal masker that is closest to the central 
    # to every critical bandrate and adds up all the non tonals in that bin
    # into this one bin, and gives a marker of 2 to the central non tonal one
    sum1 = 0
    bark_prev = barkarray[0]
    for i in range(length):
        if(val[i]==0):
            if(barkarray[i]!=bark_prev):
                Pval[criticaldefn[bark_prev]]=10*numpy.log10(sum1)
                val[criticaldefn[bark_prev]]=2
                bark_prev = barkarray[i]
                sum1 = 0   
            Pval[i] = 0         
            sum1+=pow(10,(wave_DB[i]/10.0))
    
    # TODO -delete these
    # for i in range(len(criticaldefn)):
    #     print criticaldefn[i],Pval[criticaldefn[i]], float_barkarray[criticaldefn[i]],quiet_threshold[criticaldefn[i]]

    # removes the invalid tonal and non tonal markers
    # removes the ones below threshold of hearing and 
    # removes the smaller among the ones within 0.5 barks to each other
    count = 0
    for i in range(length):
        if(val[i]==1 or val[i]==2):
            if(Pval[i]<quiet_threshold[i]):
                val[i] = 0
                Pval[i]= 0
            if(count>0):
                if(float_barkarray[valprev]-float_barkarray[i]<0.5):
                    if(Pval[valprev]>Pval[i]):
                        Pval[i] = 0
                        val[i]  = 0
                    else:
                        Pval[valprev]=0
                        val[valprev]=0
            count = count+1
            valprev = i
    return Pval,val
            
def compute_masking_indices(val,bark_array_float,length):
    masking_indices  = numpy.empty(length)
    for i in range(length):
        if(val[i]==1):
           masking_indices[i]=((-6.025)-(0.275*bark_array_float[i]))
        elif(val[i]==2):
           masking_indices[i]=((-2.025)-(0.175*bark_array_float[i]))
    return masking_indices
    # 1 -2.17379224196
    # 3 -2.46688387311
    # 5 -2.74663618721
    # 8 -3.12766989924
    # 14 -3.7284959941
    # 21 -4.20338323862
    # 39 -4.87049645039
    # 56 -5.23291612556
    # 81 -5.61140135356
    # 157 -6.15723003434


def spreading_function(zmaskee,zmasker,Pmasker):
    z = zmaskee-zmasker
    c = 0
    if ((-3 <= z) and (z<-1)):
        v = (17*z)-(0.4*Pmasker)+11
        c = 1
    elif ((-1<=z) and (z<0)):
        v = ((0.4*Pmasker)+6)*z
        c = 2
    elif ((0<=z) and (z<1)):
        v = (-17)*z
        c = 3
    elif ((1<=z) and (z<8)):
        v = ((-17)*z)+(0.15*Pmasker*(z-1))
        c = 4
    if c==0:
        return -1
    if(c==1 or c==2 or c==3 or c==4):
        return 10**(v/10.0)

def tonal_nontonal_threshold(Pval,val,float_barkarray_val,mask_index,float_barkarray,length):
    sum = 0
    for i in range(length):
        if (val[i] ==2 or val[i] ==1):
            val_spreading_fn = spreading_function(float_barkarray_val,float_barkarray[i],Pval[i])
            if(val_spreading_fn!=-1):
                l = Pval[i]+mask_index[i]+val_spreading_fn
                x = 10**(l/10.0)
                sum= sum+x
    return sum

def globalMaskingThreshold(Pval,val,float_barkarray,quiet_threshold,mask_index,length):
    global_masker = numpy.empty(length)
    for i in range(length):
        x = tonal_nontonal_threshold(Pval,val,float_barkarray[i],mask_index,float_barkarray,length)
        global_masker[i]=10*numpy.log10((10**(quiet_threshold[i]/10))+x)
    return global_masker

def minMaskingThreshold(global_mask,length):
    Subband_size = length / N_SUBBAND
    min_masker = numpy.empty(N_SUBBAND)
    # print length
    # print Subband_size
    # print N_SUBBAND
    for i in range(N_SUBBAND):
        min_masker[i] = global_mask[Subband_size*i]
        for j in range(Subband_size):
                if (min_masker[i] > global_mask[(Subband_size*i)+j]):
                    min_masker[i] = global_mask[(Subband_size*i)+j]
    return min_masker
    

def watermarking_block(signal,bits,Fs,Win,Step):
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

    frequency_array = []
    for i in range((Win/2)+1):
        frequency_array.append((i*Fs)/float(Win))
    frequency_array = numpy.array(frequency_array)
    threshold_quiet_array = threshold_quiet(frequency_array,((Win/2)+1))
    bark_array_float = bark(frequency_array)
    bark_array = numpy.array(bark_array_float,dtype=int) 
    criticaldefn = numpy.array(closeto_critical(frequency_array))
    # Signal normalization
    signal = numpy.double(signal)

    # TODO uncomment this and check if results change ,only at the end though
    # signal = signal / (2.0 ** 15)
    # DC = signal.mean()
    # MAX = (numpy.abs(signal)).max()
    # signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    # print 'length=',N
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        print 'countFrames',countFrames
        x       = signal[curPos:curPos+Win]                    # get current window
        curPos  = curPos + Step                           # update window position
        x1      = hann(x,Win)
        X       = (fft(x1))                                    # get fft magnitude        
        #TODO --> separate function
        Xabs    = abs(X)
        # normalize fft
        Xabslog = 10*numpy.log10(numpy.square(Xabs))
        #TODO - examine how needed this is
        Xabslog_norm = SPL_normalise(Xabslog,Win)
       
        P,v = tonal_markers_sound(Xabslog_norm[:(Win/2)+1],(Win/2)+1,bark_array,bark_array_float,criticaldefn,threshold_quiet_array)
        mask_indices = compute_masking_indices(v,bark_array_float,len(v))
        
        global_mask = globalMaskingThreshold(P,v,bark_array_float,threshold_quiet_array,mask_indices,len(v))
        
        min_mask = minMaskingThreshold(global_mask,len(v))
        
    
    # # print 'countFrames=',countFrames
    # return numpy.array(stFeatures)