from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import host_subplot
from numpy import zeros, newaxis
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np
###

foldername = 'D:/4-5th semester -MasterThesisDLR/files/Analysis/Simu7_img1line14_DSM/';
ii=0

# Read Optimized lines(X,Y,Z)
# Read Optimized lines(X,Y,Z) # original
try:
	XYZ = np.loadtxt((foldername + 'Poly3D_0.txt'), skiprows=1, usecols=(0, 1, 2), unpack=True)
except IOError as e:
	print('Poly3D_0.txt not found')
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
	radii1 = np.sqrt(np.diag(A))
	B = Sigmaxxhat[3:6,3:6,indx]#posterior
	radii2 = np.sqrt(np.diag(B))
	sigmaX_ = np.append(sigmaX_,np.sqrt((radii1[0]**2+radii2[0]**2)/2))
	sigmaY_ = np.append(sigmaY_,np.sqrt((radii1[1]**2+radii2[1]**2)/2))
	sigmaZ_ = np.append(sigmaZ_,np.sqrt((radii1[2]**2+radii2[2]**2)/2))

	A = Sigmaxx[0:3,0:3,indx]#priori
	radii = np.sqrt(np.diag(A))
	B = Sigmaxx[3:6,3:6,indx]#priori
	radii2 = np.sqrt(np.diag(B))
	sigmaX = np.append(sigmaX,np.sqrt((radii1[0]**2+radii2[0]**2)/2))
	sigmaY = np.append(sigmaY,np.sqrt((radii1[1]**2+radii2[1]**2)/2))
	sigmaZ = np.append(sigmaZ,np.sqrt((radii1[2]**2+radii2[2]**2)/2))
sigmaXY_ = np.sqrt(sigmaX_**2+sigmaY_**2)
sigmaXY = np.sqrt(sigmaX**2+sigmaY**2)


## Plot 1: basics

host = host_subplot(111,axes_class=AA.Axes);
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
#par2 = host.twinx()
par3 = host.twinx()

new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
par1.axis["right"].toggle(all=True)

#offset2 = 60
#new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset2, 0))
#par2.axis["right"].toggle(all=True)

offset3 = 60
new_fixed_axis = par3.get_grid_helper().new_fixed_axis
par3.axis["right"] = new_fixed_axis(loc="right", axes=par3, offset=(offset3, 0))
par3.axis["right"].toggle(all=True)

host.set_xlabel('the i$^{th}$ node')

host.set_ylabel('amount of images')
par1.set_ylabel('redundancies')
#par2.set_ylabel('posterior standard deviation $\hat{\sigma}_0$ [pixel]')
par3.set_ylabel('height of the reconstructed line nodes [meter]')

host.set_ylim([0,y1.max()+1])
host.set_xlim([1,x.max()])

p1, = host.plot(x, y1, 'b-', label='amount of images',marker='s',markeredgewidth=0,markersize=4)
p2, = par1.plot(x, y2, 'g-', label='redundancies',marker='o',markeredgewidth=0,markersize=4)
#p3, = par2.plot(x, y3, 'r-', label='posterior standard deviation',marker='D',markeredgewidth=0,markersize=4)
p4, = par3.plot(x, XYZ[2,1:], 'k-', label='height of reconstructed node',marker='v',markeredgewidth=0,markersize=6)

host.legend(loc='lower left', bbox_to_anchor=(0.6,-0.33),fontsize=9)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
#par2.axis["right"].label.set_color(p3.get_color())
par3.axis["right"].label.set_color(p4.get_color())

plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_ImgNum.png'), bbox_inches="tight", dpi=300)



plt.clf()

## Plot 2: posterior Variance-Covariance 

host = host_subplot(111,axes_class=AA.Axes);
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
par1.axis["right"].toggle(all=True)

offset2 = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset2, 0))
par2.axis["right"].toggle(all=True)

host.set_xlabel('the i$^{th}$ node')

host.set_ylabel('amount of images')
par1.set_ylabel('posterior variances of estimated parameters $\hat{\Sigma}_{\hat{X}\hat{X}}$ [meter]') # which notation better
par2.set_ylabel('posterior standard deviation $\hat{\sigma}_0$ [pixel]')

host.set_ylim([0,y1.max()+1])
host.set_xlim([1,x.max()])

p1, = host.plot(x, y1, label='amount of images',marker='s',markeredgewidth=0,markersize=4)

p2, = par1.plot(x, sigmaXY_,'-', color='lime', label='$\sqrt{\hat{\sigma}_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$',marker='o',markeredgewidth=0,markersize=4)
p2, = par1.plot(x, sigmaZ_,'-', color='green', label='$\hat{\sigma}_\hat{Z}$',marker='^',markeredgewidth=0,markersize=6)

p3, = par2.plot(x, y3, label='posterior standard deviation', color='red',marker='D',markeredgewidth=0,markersize=4)

host.legend(loc='lower left', bbox_to_anchor=(0.6,-0.36),fontsize=9)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())


plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_SigmaXXhat.png'), bbox_inches="tight", dpi=300)



plt.clf()

## Plot 3: priori Variance-Covariance 

host = host_subplot(111,axes_class=AA.Axes);
plt.subplots_adjust(right=0.75)

par1 = host.twinx()

new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
par1.axis["right"].toggle(all=True)


host.set_xlabel('the i$^{th}$ node')

host.set_ylabel('amount of images')
par1.set_ylabel('priori variances of estimated parameters [meter]') # which notation better

host.set_ylim([0,y1.max()+1])
host.set_xlim([1,x.max()])

p1, = host.plot(x, y1, label='amount of images',marker='s',markeredgewidth=0,markersize=4)

p2, = par1.plot(x, sigmaXY,'-', color='gold', label='$\sqrt{\sigma_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$',marker='o',markeredgewidth=0,markersize=4)
p2, = par1.plot(x, sigmaZ,'-', color='darkorange', label='$\sigma_\hat{Z}$',marker='^',markeredgewidth=0,markersize=6)

host.legend(loc='lower left', bbox_to_anchor=(-0.05,-0.36),fontsize=9)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_SigmaXX.png'), bbox_inches="tight", dpi=300)



plt.clf()


## Plot 4

host = host_subplot(111,axes_class=AA.Axes);
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()

new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
par1.axis["right"].toggle(all=True)

offset2 = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset2, 0))
par2.axis["right"].toggle(all=True)

offset3 = 120
new_fixed_axis = par3.get_grid_helper().new_fixed_axis
par3.axis["right"] = new_fixed_axis(loc="right", axes=par3, offset=(offset3, 0))
par3.axis["right"].toggle(all=True)

host.set_xlabel('the i$^{th}$ node')

host.set_ylabel('amount of images')
par1.set_ylabel('posterior variances of estimated parameters $\sqrt{\hat{\sigma}_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$ [meter]') # which notation better
par2.set_ylabel('posterior variances of estimated parameters $\hat{\sigma}_{\hat{Z}}$ [meter]') # which notation better
par3.set_ylabel('posterior standard deviation $\hat{\sigma}_0$ [pixel]')

host.set_ylim([0,y1.max()+1])
host.set_xlim([1,x.max()])

p1, = host.plot(x, y1, label='amount of images',marker='s',markeredgewidth=0,markersize=4)

p2, = par1.plot(x, sigmaXY_,'-', color='lime', label='$\sqrt{\hat{\sigma}_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$',marker='o',markeredgewidth=0,markersize=4)
p3, = par2.plot(x, sigmaZ_,'-', color='green', label='$\hat{\sigma}_\hat{Z}$',marker='^',markeredgewidth=0,markersize=6)

p4, = par3.plot(x, y3, label='posterior standard deviation', color='red',marker='D',markeredgewidth=0,markersize=4)

host.legend(loc='lower left', bbox_to_anchor=(0.6,-0.36),fontsize=9)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())
par3.axis["right"].label.set_color(p4.get_color())


plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_XY_Z_hat.png'), bbox_inches="tight", dpi=300)



plt.clf()


## Plot 5

host = host_subplot(111,axes_class=AA.Axes);
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
par1.axis["right"].toggle(all=True)

offset2 = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset2, 0))
par2.axis["right"].toggle(all=True)

host.set_xlabel('the i$^{th}$ node')

host.set_ylabel('amount of images')
par1.set_ylabel('priori variances of estimated parameters $\sqrt{\sigma_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$ [meter]') # which notation better
par2.set_ylabel('priori variances of estimated parameters $\sigma_{\hat{Z}}$ [meter]') # which notation better


host.set_ylim([0,y1.max()+1])
host.set_xlim([1,x.max()])

p1, = host.plot(x, y1, label='amount of images',marker='s',markeredgewidth=0,markersize=4)

p2, = par1.plot(x, sigmaXY,'-', color='gold', label='$\sqrt{\sigma_\hat{X}^2+\hat{\sigma}_\hat{Y}^2}$',marker='o',markeredgewidth=0,markersize=4)
p3, = par2.plot(x, sigmaZ,'-', color='darkorange', label='$\sigma_\hat{Z}$',marker='^',markeredgewidth=0,markersize=6)


host.legend(loc='lower left', bbox_to_anchor=(-0.05,-0.36),fontsize=9)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())


plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(10, 7)
plt.savefig((foldername+'Simu_XY_Z.png'), bbox_inches="tight", dpi=300)



plt.clf()

####### plot 6: correlation


fig1 = plt.figure(1)

# plt.subplot(121)
plt.scatter(y1,sigmaXY,	c=sigmaXY,marker='o')


plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(6, 8)
#plt.tight_layout()
plt.savefig((foldername+'Simu_ImgNumSigmaXXCorrelation.png'), bbox_inches="tight", dpi=300)

