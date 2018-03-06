from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import host_subplot
from numpy import zeros, newaxis
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np
###

foldername = '/home/sheu_ch/la/A9/Analysis/Simu_0306/';
ii=0


# Read Optimized lines(X,Y,Z)
try:
	XYZ = np.loadtxt((foldername + 'Poly3D_'+str(ii)+'.txt'), skiprows=1, usecols=(0, 1, 2), unpack=True)
except IOError as e:
	print('Poly3D_'+str(ii)+'.txt not found')
numOpt = XYZ.shape[1]

## Read Statistics
totalimgnum = np.empty((0),int);
totalpointnum = np.empty((0),int);
redundancy = np.empty((0),int);
sigma0hat = np.empty((0),float);
Sigmaxxhat = np.empty((6,6,0),float);
Sigmaxx = np.empty((6,6,0),float);

with open((foldername+'statistics_'+str(ii)+'.txt'),'r') as fid:
	for line in fid:
		if 'totalimgnum' in line:
			data = line.split()
			totalimgnum = np.append(totalimgnum, int(data[1]))
		elif 'redundancy' in line:
			data = line.split()
			redundancy = np.append(redundancy,int(data[1]))
		elif 'sigma0hat' in line:
			data = line.split()
			sigma0hat = np.append(sigma0hat,float(data[1]))
		elif 'Sigmaxxhat' in line:
			line = fid.readline()
			data = line.split()
			temp1 = np.array([[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])]])
			for i in range(1,6):
				line = fid.readline()
				data = line.split()
				temp2 = np.array([[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])]])
				temp1 = np.append(temp1,temp2,axis=0)
			temp1 = temp1[:,:,newaxis]
			Sigmaxxhat = np.append(Sigmaxxhat, temp1, axis=2)
		elif 'Sigmaxx' in line:
			line = fid.readline()
			data = line.split()
			temp1 = np.array([[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])]])
			for i in range(1,6):
				line = fid.readline()
				data = line.split()
				temp2 = np.array([[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])]])
				temp1 = np.append(temp1,temp2,axis=0)
			temp1 = temp1[:,:,newaxis]
			Sigmaxx = np.append(Sigmaxx, temp1, axis=2)
fid.close()
num = totalimgnum.shape[0]

x = np.arange(1,num+1)
y1 = totalimgnum
y2 = redundancy
y3 = sigma0hat

sigmaX_ = np.empty((0),float) #posterior
sigmaY_ = np.empty((0),float)
sigmaZ_ = np.empty((0),float)
sigmaXY_ = np.empty((0),float)

sigmaX = np.empty((0),float) #priori
sigmaY = np.empty((0),float)
sigmaZ = np.empty((0),float)
sigmaXY = np.empty((0),float)
for indx in range(num):
	A = Sigmaxxhat[0:3,0:3,indx]#posterior
	B = Sigmaxxhat[3:6,3:6,indx]#posterior
	sigmaX_ = np.append(sigmaX_,np.sqrt((A[0,0]+B[0,0])/2))
	sigmaY_ = np.append(sigmaY_,np.sqrt((A[1,1]+B[1,1])/2))
	sigmaZ_ = np.append(sigmaZ_,np.sqrt((A[2,2]+B[2,2])/2))

	A = Sigmaxx[0:3,0:3,indx]#priori
	B = Sigmaxx[3:6,3:6,indx]#priori
	sigmaX = np.append(sigmaX,np.sqrt((A[0,0]+B[0,0])/2))
	sigmaY = np.append(sigmaY,np.sqrt((A[1,1]+B[1,1])/2))
	sigmaZ = np.append(sigmaZ,np.sqrt((A[2,2]+B[2,2])/2))
	
sigmaXY_ = np.sqrt(sigmaX_**2+sigmaY_**2)
sigmaXY = np.sqrt(sigmaX**2+sigmaY**2)


## Plot 1: information on image amount, redundancies and the reconstructed height

f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x, y1, 'b-', label='amount of images',marker='s',markeredgewidth=0,markersize=4)
axarr[0].set_ylim([0,y1.max()+1])
axarr[0].legend(loc='lower right',fontsize=9)
axarr[0].grid(True)

axarr[1].plot(x, y2, 'g-', label='redundancies',marker='o',markeredgewidth=0,markersize=4)
axarr[1].legend(loc='lower right',fontsize=9)
axarr[1].grid(True)

axarr[2].plot(x, XYZ[2,1:], 'k-', label='height of reconstructed line node',marker='v',markeredgewidth=0,markersize=6)
axarr[2].set_ylabel('[meter]')
axarr[2].legend(loc='upper right',fontsize=9)
axarr[2].grid(True)

axarr[2].set_xlabel('the i$^{th}$ node')
axarr[2].set_xlim([1,x.max()])

plt.gcf().set_size_inches(10, 6)
plt.savefig((foldername+'Simu_ImgNum.png'), bbox_inches="tight", dpi=300)

plt.clf()


## Plot 2: posterior Variance-Covariance 

f, axarr = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[2,2,3]})

axarr[0].plot(x, y1, 'b-', label='amount of images',marker='s',markeredgewidth=0,markersize=4)
axarr[0].set_ylim([0,y1.max()+1])
axarr[0].legend(loc='lower left',fontsize=9)
axarr[0].grid(True)

axarr[1].plot(x, y3, label='posterior STD of the measurements, $\hat{\sigma}_0$', color='red',marker='D',markeredgewidth=0,markersize=4)
axarr[1].set_ylim([y3.min()-0.05,y3.max()+0.05])
axarr[1].set_ylabel('[pixel]')
axarr[1].legend(loc='upper left',fontsize=9)
axarr[1].grid(True)

axarr[2].plot(x, sigmaXY_,'-', color='lime', label='posterior STD of the unknowns in horizontal direction, $\sqrt{\hat{\sigma}_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$',marker='o',markeredgewidth=0,markersize=4)
axarr[2].plot(x, sigmaZ_,'-', color='green', label='posterior STD of the unknowns in vertical direction, $\hat{\sigma}_\hat{Z}$',marker='^',markeredgewidth=0,markersize=6)
axarr[2].set_ylim([0,sigmaZ_.max()+0.005])
axarr[2].set_ylabel('[meter]')
axarr[2].legend(loc='best',fontsize=9)
axarr[2].grid(True)

axarr[2].set_xlabel('the i$^{th}$ node')
axarr[2].set_xlim([1,x.max()])

plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_SigmaXXhat.png'), bbox_inches="tight", dpi=300)

plt.clf()


## Plot 3: priori Variance-Covariance 

f, axarr = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[2,2,3]})

axarr[0].plot(x, y1, 'b-', label='amount of images',marker='s',markeredgewidth=0,markersize=4)
axarr[0].set_ylim([0,y1.max()+1])
axarr[0].legend(loc='lower right',fontsize=9)
axarr[0].grid(True)

axarr[1].plot(x, y3, label='posterior STD of the measurements, $\hat{\sigma}_0$', color='red',marker='D',markeredgewidth=0,markersize=4)
axarr[1].set_ylim([y3.min()-0.08,y3.max()+0.05])
axarr[1].set_ylabel('[pixel]')
axarr[1].legend(loc='lower right',fontsize=9)
axarr[1].grid(True)

axarr[2].plot(x, sigmaXY,'-', color='gold', label='priori STD of the unknowns in horizontal direction, $\sqrt{\sigma_\hat{X}^2+\sigma_\hat{Y}^2}$',marker='o',markeredgewidth=0,markersize=4)
axarr[2].plot(x, sigmaZ,'-', color='darkorange', label='priori STD of the unknowns in vertical direction, $\sigma_\hat{Z}$',marker='^',markeredgewidth=0,markersize=6)
axarr[2].set_ylim([sigmaXY.min()-0.005,sigmaZ.max()+0.01])
axarr[2].set_ylabel('[meter]')
axarr[2].legend(loc='upper right',fontsize=9)
axarr[2].grid(True)

axarr[2].set_xlabel('the i$^{th}$ node')
axarr[2].set_xlim([1,x.max()])

plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_SigmaXX.png'), bbox_inches="tight", dpi=300)

plt.clf()


####### plot 6: correlation


fig1 = plt.figure(1)

# plt.subplot(121)
plt.scatter(y1,sigmaXY,	c=sigmaXY,marker='o')


plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(6, 8)
#plt.tight_layout()
plt.savefig((foldername+'Simu_ImgNumSigmaXXCorrelation.png'), bbox_inches="tight", dpi=300)



plt.clf()



