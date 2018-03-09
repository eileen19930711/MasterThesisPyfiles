from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

#foldername = 'D:/4-5th semester -MasterThesisDLR/files/Analysis/Simu_0306/';
foldername = '/home/sheu_ch/la/A9/Analysis/Simu_0306/';
ii=0



# Read Optimized lines(X,Y,Z) # original
try:
	XYZ = np.loadtxt((foldername + 'Poly3D_0.txt'), skiprows=1, usecols=(0, 1, 2), unpack=True)
except IOError as e:
	print('Poly3D_0.txt not found')
numOpt = XYZ.shape[1]

# Read Optimized_Unrefined lines(X,Y,Z) # resampled
try:
	XYZ_DSM = np.loadtxt((foldername + 'dist_opti_proj_' + str(ii) + '.txt'), usecols=(3, 4, 5), unpack=True)
	XYZ_dist = np.loadtxt((foldername + 'dist_opti_proj_' + str(ii) + '.txt'), usecols=(6), unpack=True)
except IOError as e:
	print('file dist_opti_proj_'+str(i)+'.txt not found')

numUnr = XYZ_DSM.shape[1]
print(np.max(np.fabs(XYZ_dist)))

# Read Optimized_True lines(X,Y,Z) # resampled
try:
	XYZ_true = np.loadtxt((foldername + 'dist_opti_true_' + str(ii) + '.txt'), usecols=(3, 4, 5), unpack=True)
	XYZ_dist_true = np.loadtxt((foldername + 'dist_opti_true_' + str(ii) + '.txt'), usecols=(6), unpack=True)
except IOError as e:
	print('file dist_opti_true_'+str(i)+'.txt not found')

numTru = XYZ_true.shape[1]


truelength = ((XYZ_true[0,0]-XYZ_true[0,-1])**2+(XYZ_true[1,0]-XYZ_true[1,-1])**2+(XYZ_true[2,0]-XYZ_true[2,-1])**2)**(0.5)
print(truelength)

####### figure 1: Reconstruction_3D

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
ax1.plot(XYZ[0,0:],XYZ[1,0:],XYZ[2,0:],'r',marker='o',markeredgewidth=0,markersize=4,label='reconstructed line segment');


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
#plt.tight_layout()
plt.savefig((foldername+'Simu_3D_1.png'), bbox_inches="tight", dpi=300)





####### figure 2: Reconstruction_3D


fig2 = plt.figure(2);
ax2 = fig2.add_subplot(111,projection='3d');

# projection on 3 planes
ax2.plot(np.ones(numOpt)*(minX-0.5),	XYZ[1,0:],	XYZ[2,0:],	color='indianred',marker='o',markeredgewidth=0,markersize=4)
ax2.plot(np.ones(numTru)*(minX-0.5),	XYZ_true[1,0:],	XYZ_true[2,0:],	color='steelblue')

ax2.plot(XYZ[0,0:],	np.ones(numOpt)*(maxY+0.5),	XYZ[2,0:],	color='indianred',marker='o',markeredgewidth=0,markersize=4)
ax2.plot(XYZ_true[0,0:],	np.ones(numTru)*(maxY+0.5),	XYZ_true[2,0:],	color='steelblue')

ax2.plot(XYZ[0,0:],	XYZ[1,0:],	np.ones(numOpt)*(minZ-0.5),	color='indianred',marker='o',markeredgewidth=0,markersize=4)
ax2.plot(XYZ_true[0,0:],	XYZ_true[1,0:],	np.ones(numTru)*(minZ-0.5),	color='steelblue')

# optimized
ax2.plot(XYZ[0,0:],XYZ[1,0:],XYZ[2,0:],'r-',marker='o',markeredgewidth=0,markersize=4,label='reconstructed line segment');

# DSM profile; before optimization
ax2.plot(XYZ_true[0,0:],XYZ_true[1,0:],XYZ_true[2,0:],'b-',label='true line segment');


ax2.set_xlim(minX-0.5, maxX+0.5)
ax2.set_ylim(minY-0.5, maxY+0.5)
ax2.set_zlim(minZ-0.5, maxZ+0.5)
#ax2.set_title('Reconstructed line segment')#, fontsize=10)
ax2.set_xlabel("X coordinate [m]")
ax2.set_ylabel("Y coordinate [m]")
ax2.set_zlabel("Z coordinate [m]")
ax2.legend(loc='lower left', bbox_to_anchor=(0.05,-0.2),fontsize=9)
plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(11, 8)
#plt.tight_layout()
plt.savefig((foldername+'Simu_3D_2.png'), bbox_inches="tight", dpi=300)





####### figure 3: DifferencesHistogram

fig3 = plt.figure(3);
ax3 = fig3.add_subplot(111);
# the histogram of the data
n, bins, patches = plt.hist(XYZ_dist, 50, normed=1, facecolor='mediumaquamarine', alpha=0.75,label='data: differences between the reconstructed line and the approximate line')

# add a 'best fit' line
y = mlab.normpdf( bins, np.mean(XYZ_dist), np.std(XYZ_dist))
plt.plot(bins, y, 'b-', linewidth=1,label='normal probability density function')
plt.text(np.mean(XYZ_dist)+np.std(XYZ_dist)/2, y.max(axis=0), '$\mathcal{N}(%.3f,%.3f^2)$' %(np.mean(XYZ_dist),np.std(XYZ_dist)),color='blue')
#ax3.set_title('Histogram of the differences between the unrefined DSM profile and the reconstructed 3D line segment')#, fontsize=10)
ax3.set_xlabel('Distance from the reconstructed line to the approximate line [m]')
#ax3.set_ylabel('amount')
ax3.legend(loc='lower left', bbox_to_anchor=(0,-0.65),fontsize=9)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.subplots_adjust(bottom=0.5)
plt.gcf().set_size_inches(6, 4)
plt.savefig((foldername+'Simu_hist_1.png'), bbox_inches="tight", dpi=300)





####### figure 4: DifferencesHistogram

fig4 = plt.figure(4);
ax4 = fig4.add_subplot(111);
# the histogram of the data
n, bins, patches = plt.hist(XYZ_dist_true, 20, normed=1, facecolor='mediumaquamarine', alpha=0.75,label='data: differences between reconstructed line and true line')

# add a 'best fit' line
y = mlab.normpdf( bins, np.mean(XYZ_dist_true), np.std(XYZ_dist_true))
plt.plot(bins, y, 'b-', linewidth=1,label='normal probability density function')
plt.text(np.mean(XYZ_dist_true)+np.std(XYZ_dist_true)/0.8, y.max(axis=0), '$\mathcal{N}(%.3f,%.3f^2)$' %(np.mean(XYZ_dist_true),np.std(XYZ_dist_true)),color='blue')

#ax4.set_title('Histogram of the differences between the unrefined DSM profile and the reconstructed 3D line segment')#, fontsize=10)
ax4.set_xlabel('Distance from the reconstructed line to the true line [m]')
#ax4.set_ylabel('amount')
ax4.legend(loc='lower left', bbox_to_anchor=(0,-0.65),fontsize=9)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.subplots_adjust(bottom=0.5)
plt.gcf().set_size_inches(6, 4)
plt.savefig((foldername+'Simu_hist_2.png'), bbox_inches="tight", dpi=300)


#######
#plt.show();


