from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

foldername = 'D:/4-5th semester -MasterThesisDLR/files/Analysis/Test_img1line5/';
ii=4

# Read Optimized lines(X,Y,Z) # original
try:
	XYZ = np.loadtxt((foldername + 'Poly3D_'+str(ii)+'.txt'), skiprows=1, usecols=(0, 1, 2), unpack=True)
except IOError as e:
	print('Poly3D_'+str(ii)+'.txt not found')
numOpt = XYZ.shape[1]

# Read Optimized_Unrefined lines(X,Y,Z) # resampled
try:
	XYZ_DSM = np.loadtxt((foldername + 'dist_opti_proj_' + str(ii) + '.txt'), usecols=(3, 4, 5), unpack=True)
	XYZ_dist = np.loadtxt((foldername + 'dist_opti_proj_' + str(ii) + '.txt'), usecols=(6), unpack=True)
except IOError as e:
	print('file dist_opti_proj_'+str(i)+'.txt not found')

numUnr = XYZ_DSM.shape[1]
print(np.max(np.fabs(XYZ_dist)))



####### figure 1: ReconstructionBeforeAfter_3D

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111,projection='3d')

minX = min(XYZ[0,0:].min(), XYZ_DSM[0, 0:].min())
maxX = min(XYZ[0,0:].max(), XYZ_DSM[0, 0:].max())
minY = min(XYZ[1,0:].min(), XYZ_DSM[1, 0:].min())
maxY = max(XYZ[1,0:].max(), XYZ_DSM[1, 0:].max())
minZ = min(XYZ[2,0:].min(), XYZ_DSM[2, 0:].min())
maxZ = max(XYZ[2,0:].max(), XYZ_DSM[2, 0:].max())

# projection on 3 planes
ax1.plot(np.ones(numOpt)*(minX-0.5),	XYZ[1,0:],	XYZ[2,0:],linewidth=0.5,	color='indianred',marker='o',markeredgewidth=0,markersize=2)
ax1.plot(np.ones(10)*(minX-0.5),	XYZ[1,12:22],	XYZ[2,12:22],	color='indianred',marker='o',markeredgewidth=0,markersize=4)
#ax1.plot(np.ones(numUnr) * (minX-0.5), XYZ_DSM[1, 0:], XYZ_DSM[2, 0:], color='gray')

ax1.plot(XYZ[0,0:],	np.ones(numOpt)*(maxY+0.5),	XYZ[2,0:],linewidth=0.5,	color='indianred',marker='o',markeredgewidth=0,markersize=2)
ax1.plot(XYZ[0,12:22],	np.ones(10)*(maxY+0.5),	XYZ[2,12:22],	color='indianred',marker='o',markeredgewidth=0,markersize=4)
#ax1.plot(XYZ_DSM[0, 0:], np.ones(numUnr) * (maxY + 0.5), XYZ_DSM[2, 0:], color='gray')

ax1.plot(XYZ[0,0:],	XYZ[1,0:],	np.ones(numOpt)*(minZ-0.5),linewidth=0.5,	color='indianred',marker='o',markeredgewidth=0,markersize=2)
ax1.plot(XYZ[0,12:22],	XYZ[1,12:22],	np.ones(10)*(minZ-0.5),	color='indianred',marker='o',markeredgewidth=0,markersize=4)
#ax1.plot(XYZ_DSM[0, 0:], XYZ_DSM[1, 0:], np.ones(numUnr) * (minZ - 0.5), color='gray')

# DSM profile; before optimization
#ax1.plot(XYZ_DSM[0, 0:], XYZ_DSM[1, 0:], XYZ_DSM[2, 0:], 'k-', label='approximate line segment');

# optimized
ax1.plot(XYZ[0,0:],XYZ[1,0:],XYZ[2,0:],'r',linewidth=0.5,marker='o',markeredgewidth=0,markersize=2,label='reconstructed line segment')
ax1.plot(XYZ[0,12:22],XYZ[1,12:22],XYZ[2,12:22],'r',marker='o',markeredgewidth=0,markersize=4,label='reconstructed line segment')


###### axis.equal #####
## Create cubic bounding box to simulate equal aspect ratio
#max_range = np.array([Lines3D[0:,0].max()-Lines3D[0:,0].min(),
#			Lines3D[0:,1].max()-Lines3D[0:,1].min(), 
#			Lines3D[0:,2].max()-Lines3D[0:,2].min()]).max()
#Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(Lines3D[0:,0].max()+Lines3D[0:,0].min())
#Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Lines3D[0:,1].max()+Lines3D[0:,1].min())
#Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Lines3D[0:,2].max()+Lines3D[0:,2].min())
## Comment or uncomment following both lines to test the fake bounding box:
#for xb, yb, zb in zip(Xb, Yb, Zb):
#	ax.plot([xb], [yb], [zb], 'w')

#ax.set_xlim([691200.568959, 691205.331189])
#ax.set_ylim([5383692.144951, 5383709.486125])
########################

ax1.set_xlim(minX-0.5, maxX+0.5)
ax1.set_ylim(minY-0.5, maxY+0.5)
ax1.set_zlim(minZ-0.5, maxZ+0.5)
#ax1.set_title('Reconstructed line segment')#, fontsize=10)
ax1.set_xlabel("X coordinate [m]")
ax1.set_ylabel("Y coordinate [m]")
ax1.set_zlabel("Z coordinate [m]")
ax1.legend(loc='lower left', bbox_to_anchor=(0.05,-0.2),fontsize=9)
plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(11, 8)
plt.savefig((foldername+'Test_3D_specific.png'), bbox_inches="tight", dpi=300)


#######
#plt.show();


