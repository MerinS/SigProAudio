# parity checks - 1,2,3
# check no - bits checked 
# 1 - 1,2,3,4
# 2 - 3,4,6
# 3 - 1,4,5
# checks going wrong - bit to flip
# TODO Automate this process
# 1 - 2
# 2 - 6
# 3 - 5
# 1,2 - 3
# 1,3 - 1
# 2,3 - 4
# 1,2,3 - 2,5,6
check_matrix  = [[1,2,3,4],[3,4,6],[1,4,5]]
def parity_check(values):
	index = []
	for i in range(len(check_matrix)):
		sum = 0
		for j in range(len(check_matrix[i])):
			sum += values[check_matrix[i][j]-1]
		if(sum%2==0):
			index.append(0)
		else:
 			index.append(1)
	return index

def bit_flip(values,bits):
	for i in range(len(bits)):
		if(values[bits[i]-1]):
			values[bits[i]-1] = 0
		elif(values[bits[i]-1]==0):
			values[bits[i]-1] = 1
	return values

# solutions obtained by analyzing the parity check matrices
# HardCoded
# TODO - automate this process
codewords = [[0,0,0,0,0,0,0],[0,0,1,1,1,0],[0,1,0,1,1,1],[0,1,1,0,0,1],[1,0,0,1,0,1],[1,0,1,0,1,1],[1,1,0,0,1,0],[1,1,1,1,0,0]]
infowords = ['000','001','010','011','100','101','110','111']
def codewordtoinfo(values):
	if(values in codewords):
		return infowords[codewords.index(values)]


def LDPC_decode(watermark):
	info_seq = str()
	for i in range(len(watermark)/6):
	 	values = watermark[(i*6):(i*6)+6]
        	index  = parity_check(values)
		dec_value = 4*index[0] + 2*index[1] + index[0]
		if(dec_value==0):
			values = bit_flip(values,[2,6,5])
		elif(dec_value==1):
			values = bit_flip(values,[5])
		elif(dec_value==2):
			values = bit_flip(values,[6])
		elif(dec_value==3):
			values = bit_flip(values,[4])
		elif(dec_value==4):
		    values = bit_flip(values,[2])
		elif(dec_value==5):
			values = bit_flip(values,[1])
		elif(dec_value==6):
			values = bit_flip(values,[3])
        	info_value = codewordtoinfo(values)
        info_seq = info_seq + info_value
    return info_seq