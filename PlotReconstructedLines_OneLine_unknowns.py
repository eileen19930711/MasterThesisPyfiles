from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

#foldername = 'D:/4-5th semester -MasterThesisDLR/files/Analysis/Test5_img1line4_xdibias/';
foldername = '/home/sheu_ch/la/A9/Analysis/Test5_img1line4_xdibias/';
ii=3

# Read Optimized lines(X,Y,Z) # original
try:
	XYZ = np.loadtxt((foldername + 'Poly3D_'+str(ii)+'.txt'), skiprows=1, usecols=(0, 1, 2), unpack=True)
except IOError as e:
	print('Poly3D_'+str(ii)+'.txt not found')
numOpt = XYZ.shape[1]

# Read Optimized_Unrefined lines(X,Y,Z) # resampled
try:
	XYZ_DSM = np.loadtxt((foldername + 'dist_opti_proj_' + str(ii) + '.txt'), usecols=(3, 4, 5), unpack=True)
	XYZ_dist = np.loadtxt((foldername + 'dist_opti_proj_' + str(ii) + '.txt'), usecols=(6,), unpack=True)
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
ax1.plot(np.ones(numOpt)*(minX-0.5),	XYZ[1,0:],	XYZ[2,0:],	color='indianred',marker='o',markeredgewidth=0,markersize=4)
ax1.plot(np.ones(numUnr) * (minX-0.5), XYZ_DSM[1, 0:], XYZ_DSM[2, 0:], color='gray')

ax1.plot(XYZ[0,0:],	np.ones(numOpt)*(maxY+0.5),	XYZ[2,0:],	color='indianred',marker='o',markeredgewidth=0,markersize=4)
ax1.plot(XYZ_DSM[0, 0:], np.ones(numUnr) * (maxY + 0.5), XYZ_DSM[2, 0:], color='gray')

ax1.plot(XYZ[0,0:],	XYZ[1,0:],	np.ones(numOpt)*(minZ-0.5),	color='indianred',marker='o',markeredgewidth=0,markersize=4)
ax1.plot(XYZ_DSM[0, 0:], XYZ_DSM[1, 0:], np.ones(numUnr) * (minZ - 0.5), color='gray')

# DSM profile; before optimization
ax1.plot(XYZ_DSM[0, 0:], XYZ_DSM[1, 0:], XYZ_DSM[2, 0:], 'k-', label='approximate line segment');

# optimized
ax1.plot(XYZ[0,0:],XYZ[1,0:],XYZ[2,0:],'r',label='reconstructed line segment') #,marker='o',markeredgewidth=0,markersize=4


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
plt.savefig((foldername+'Test_3D.png'), bbox_inches="tight", dpi=300)


plt.clf()

####### figure 2: DifferencesHistogram

fig2 = plt.figure(2);
ax2 = fig2.add_subplot(111);
# the histogram of the data
n, bins, patches = plt.hist(XYZ_dist, 50, normed=1, facecolor='mediumaquamarine', alpha=0.75,label='data: differences between reconstructed line and DSM profile')

# add a 'best fit' line
y = mlab.normpdf( bins, np.mean(XYZ_dist), np.std(XYZ_dist))
plt.plot(bins, y, 'b-', linewidth=1,label='normal probability density function')
plt.text(np.mean(XYZ_dist)+0.03, y.max(axis=0), '$\mathcal{N}(%.3f,%.3f^2)$' %(np.mean(XYZ_dist),np.std(XYZ_dist)),color='blue')
#ax2.set_title('Histogram of the differences between the unrefined DSM profile and the reconstructed 3D line segment')#, fontsize=10)
ax2.set_xlabel('Distance from the reconstructed line to the unrefined DSM profile [m]')
#ax2.set_ylabel('amount')
ax2.legend(loc='lower left', bbox_to_anchor=(0,-0.65),fontsize=9)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.subplots_adjust(bottom=0.5)
plt.gcf().set_size_inches(6, 4)
plt.savefig((foldername+'Test_hist.png'), bbox_inches="tight", dpi=300)




#######
#plt.show();


plt.clf()


####### figure 3: ReconstructionBeforeAfter_3D

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111,projection='3d')

minX = min(XYZ[0,0:].min(), XYZ_DSM[0, 0:].min())
maxX = min(XYZ[0,0:].max(), XYZ_DSM[0, 0:].max())
minY = min(XYZ[1,0:].min(), XYZ_DSM[1, 0:].min())
maxY = max(XYZ[1,0:].max(), XYZ_DSM[1, 0:].max())
minZ = min(XYZ[2,0:].min(), XYZ_DSM[2, 0:].min())
maxZ = max(XYZ[2,0:].max(), XYZ_DSM[2, 0:].max())

# projection on 3 planes
ax1.plot(np.ones(numUnr) * (minX-0.5), XYZ_DSM[1, 0:], XYZ_DSM[2, 0:], linewidth=1, color='gray')

ax1.plot(XYZ_DSM[0, 0:], np.ones(numUnr) * (maxY + 0.5), XYZ_DSM[2, 0:], linewidth=1, color='gray')

ax1.plot(XYZ_DSM[0, 0:], XYZ_DSM[1, 0:], np.ones(numUnr) * (minZ - 0.5), linewidth=1, color='gray')

# DSM profile; before optimization
ax1.plot(XYZ_DSM[0, 0:], XYZ_DSM[1, 0:], XYZ_DSM[2, 0:], 'k-', linewidth=1, label='approximate line segment');

# ##### axis.equal #####
# # Create cubic bounding box to simulate equal aspect ratio
# max_range = np.array([XYZ_DSM[0, 0:].max()-XYZ_DSM[0, 0:].min(),
# 					  XYZ_DSM[1, 0:].max()-XYZ_DSM[1, 0:].min(),
# 					  XYZ_DSM[2, 0:].max()-XYZ_DSM[2, 0:].min()]).max()
# Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(XYZ_DSM[0, 0:].max()+XYZ_DSM[0, 0:].min())
# Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(XYZ_DSM[1, 0:].max()+XYZ_DSM[1, 0:].min())
# Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(XYZ_DSM[2, 0:].max()+XYZ_DSM[2, 0:].min())
# # Comment or uncomment following both lines to test the fake bounding box:
# for xb, yb, zb in zip(Xb, Yb, Zb):
# 	ax1.plot([xb], [yb], [zb], 'w')
#
# ax1.set_xlim([minX, maxX])
# ax1.set_ylim([minY, maxY])
# ########################

ax1.set_xlim(minX-0.5, maxX+0.5)
ax1.set_ylim(minY-0.5, maxY+0.5)
ax1.set_zlim(minZ-0.5, maxZ+0.5)

#ax1.set_title('Reconstructed line segment')#, fontsize=10)
ax1.set_xlabel("X coordinate [m]")
ax1.set_ylabel("Y coordinate [m]")
ax1.set_zlabel("Z coordinate [m]")
#ax1.legend(loc='lower left', bbox_to_anchor=(0.05,-0.2),fontsize=9)
plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(11, 8)
plt.savefig((foldername+'Test_3D_DSM.png'), bbox_inches="tight", dpi=300)


