from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib import cm
from numpy import zeros, newaxis
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

foldername = '/home/sheu_ch/la/A9/Analysis/Test_img1line1/';
ii=50

# Read Optimized lines(X,Y,Z)
with open((foldername+'Poly3D_0.txt'),'r') as fid:
	next(fid)
	data = [ map(float,line.split()) for line in fid ];
fid.closed
XYZnn = np.array(data, dtype=np.float)
XYZ = XYZnn[0:,0:3]
numOpt = XYZ.shape[0]


## Read Statistics
totalimgnum = np.empty((0),int);
#totalpointnum = np.empty((0),int);
#redundancy = np.empty((0),int);
#sigma0hat = np.empty((0),float);
Sigmaxxhat = np.empty((6,6,0),float);

with open((foldername+'statistics_0.txt'),'r') as fid:
	for line in fid:
		if 'totalimgnum' in line:
			data = line.split();
			totalimgnum = np.append(totalimgnum, int(data[1]))
#		elif 'totalpointnum' in line:
#			data = line.split();
#			totalpointnum = np.append(totalpointnum,int(data[1]));
#		elif 'redundancy' in line:
#			data = line.split();
#			redundancy = np.append(redundancy,int(data[1]));
#		elif 'sigma0hat' in line:
#			data = line.split();
#			sigma0hat = np.append(sigma0hat,float(data[1]));
#		elif 'e_mean' in line:
#			data = line.split();
#			e_mean = np.append(sigma0hat,float(data[1]));
		elif 'Sigmaxxhat' in line:
			line = fid.next();
			data = line.split();
			temp1 = np.array([[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])]])
			for i in range(1,6):
				line = fid.next();
				data = line.split();
				temp2 = np.array([[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])]])
				temp1 = np.append(temp1,temp2,axis=0);
			temp1 = temp1[:,:,newaxis]
			Sigmaxxhat = np.append(Sigmaxxhat, temp1, axis=2)
fid.close()

## Plot
fig = plt.figure();
ax = fig.add_subplot(111,projection='3d');

# measured/projected; before optimization
ax.plot(XYZ[ii:(ii+2),0],XYZ[ii:(ii+2),1],XYZ[ii:(ii+2),2],'r-');


#lambda_, v = np.linalg.eig(Sigmaxxhat[0:2,0:2,ii])
#lambda_ = np.sqrt(lambda_)*0.3 #reduce lambda_ by factor 0.3 
#ax = plt.subplot(111, aspect='equal')
#for j in xrange(1, 4):
#	ell = Ellipse(xy=(XYZ[ii,0], XYZ[ii,1]),
#		width=lambda_[0]*j*2, height=lambda_[1]*j*2,
#		angle=np.rad2deg(np.arccos(v[0, 0])))
#	ell.set_facecolor('none')
#	ax.add_artist(ell)
#plt.scatter(XYZ[ii,0], XYZ[ii,1])
#plt.show()
xx = np.empty((60,0),float);
yy = np.empty((60,0),float);
zz = np.empty((60,0),float);
ellipNumber=2
for indx in xrange(ellipNumber):
	A=Sigmaxxhat[(3*indx):(3*(indx+1)),(3*indx):(3*(indx+1)),ii]
	#A=[[9,0,0],[0,16,0],[0,0,25]]
	center=[XYZ[ii+indx,0], XYZ[ii+indx,1],XYZ[ii+indx,2]]

	#set colour map so each ellipsoid as a unique colour
	norm = colors.Normalize(vmin=0, vmax=ellipNumber)
	cmap = cm.jet
	m = cm.ScalarMappable(norm=norm, cmap=cmap)

	#find the rotation matrix and radii of the axes
	W, V = np.linalg.eig(A) # u and rotation are unitary and s is a 1-d array of singular values of A
	radii = np.sqrt(W) * 0.1 #reduce radii by factor 0.1
	print 'W' # eigenvalues
	print W
	print 'radii'
	print radii
	print 'V' # eigenvectors; rotation matrix
	print V
	print 'A'
	print A
	print 'Sigma=V*W*VT'
	print np.dot(np.dot(V,np.diag(W)),V.transpose())
	# calculate cartesian coordinates for the ellipsoid surface
	u = np.linspace(0.0, 2.0 * np.pi, 60)
	v = np.linspace(0.0, np.pi, 60)
	x = radii[0] * np.outer(np.cos(u), np.sin(v))
	y = radii[1] * np.outer(np.sin(u), np.sin(v))
	z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) *100
	print x.shape
	# rotate the ellipsoid
	for i in range(len(x)):
		for j in range(len(x)):
			[x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], V) + center
	ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=m.to_rgba(indx), linewidth=0.1, alpha=1, shade=True)
	xx = np.append(xx,x,axis=1)
	yy = np.append(yy,y,axis=1)
	zz = np.append(zz,z,axis=1)	

##### axis.equal #####
# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([xx.max()-xx.min(), yy.max()-yy.min(), zz.max()-zz.min()]).max()
print max_range
mid_x = (xx.max()+xx.min()) * 0.5
mid_y = (yy.max()+yy.min()) * 0.5
mid_z = (zz.max()+zz.min()) * 0.5
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

########################

plt.show()







#ax.set_title('Reconstructed line segments')#, fontsize=10)
#ax.set_xlabel("X coordinate [m]")
#ax.set_ylabel("Y coordinate [m]")
#ax.set_zlabel("Z coordinate [m]")
#ax.legend(loc='lower left', bbox_to_anchor=(0.05,-0.2),fontsize=12)
#plt.subplots_adjust(bottom=0.3)
#plt.gcf().set_size_inches(11, 8)
#plt.savefig((foldername+'Test_3D_1.png'), bbox_inches="tight", dpi=300)

#plt.show();














