from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

foldername = '/home/sheu_ch/la/A9/Analysis/Test4_all/'

Xmax=0
Ymax=0
Zmax=0
Xmin=999999999
Ymin=999999999
Zmin=999999999

fig = plt.figure(1);### figure 1: ReconstructionBeforeAfter
ax = fig.add_subplot(111,projection='3d');
for ii in range(0,703): #,1959)
	try:
		with open((foldername+'dist_opti_proj_'+str(ii)+'.txt'),'r') as fid:
			data = [ map(float,line.split()) for line in fid ];
		fid.closed
		XYZnn = np.array(data, dtype=np.float)
		XYZ_yhat = XYZnn[0:,0:3]
		XYZ_y = XYZnn[0:,3:6]
		XYZ_dist = XYZnn[0:,6]

		numpt = XYZ_yhat.shape[0];
		

		if ii==1:
			mylabel='DSM profile';
		else:    
			mylabel='_nolegend_'

		### optimized
		ax.plot(XYZ_yhat[0:,0],XYZ_yhat[0:,1],XYZ_yhat[0:,2],c='r',label='reconstructed lines');
		#ax.text(XYZ_yhat[-1,0],XYZ_yhat[-1,1],XYZ_yhat[-1,2]+0.2,'%d' % ii,color='b');

		#length = np.sqrt((XYZ_yhat[-1,0]-XYZ_yhat[0,0])**2+(XYZ_yhat[-1,1]-XYZ_yhat[0,1])**2)
		#if length>6:
		#	ax.plot(XYZ_yhat[0:,0],XYZ_yhat[0:,1],XYZ_yhat[0:,2],c='k',label='reconstructed lines');
		#	ax.text(XYZ_yhat[0,0],XYZ_yhat[0,1],XYZ_yhat[0,2]+0.2,'%d' % ii,color='k');


		### projected; before optimization
		ax.plot(XYZ_y[0:,0],XYZ_y[0:,1],XYZ_y[0:,2],c='k',label='DSM profile');

		if XYZ_yhat[0:,0].max()	> Xmax:
			Xmax = XYZ_yhat[0:,0].max()
		if XYZ_yhat[0:,0].min()	< Xmin:
			Xmin = XYZ_yhat[0:,0].min()	
	
		if XYZ_yhat[0:,1].max()	> Ymax:
			Ymax = XYZ_yhat[0:,1].max()
		if XYZ_yhat[0:,1].min()	< Ymin:
			Ymin = XYZ_yhat[0:,1].min()

		if XYZ_yhat[0:,2].max()	> Zmax:
			Zmax = XYZ_yhat[0:,2].max()
		if XYZ_yhat[0:,2].min()	< Zmin:
			Zmin = XYZ_yhat[0:,2].min()
		
	except IOError as e:
		print 'file dist_opti_proj_'+str(ii)+'.txt not found'


### axis.equal ### 
# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([Xmax-Xmin,Ymax-Ymin,Zmax-Zmin]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(Xmax+Xmin)
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Ymax+Ymin)
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Zmax+Zmin)
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
	ax.plot([xb], [yb], [zb], 'w')
#ax.set_xlim([690900.0, 691150.0])
#ax.set_ylim([5383850.0, 5384000.0])


plt.figure(1) 
ax.set_title('Reconstructed line segments')#, fontsize=10)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
#ax.view_init(30, 90)
#ax.legend(loc='lower left', bbox_to_anchor=(0,-0.3),fontsize=12)

plt.gcf().set_size_inches(12, 8)
ax.view_init(elev=45, azim=-75)
plt.savefig((foldername+'Reconstruction-BeforeAfter.png'), bbox_inches="tight", dpi=300)

#for angle in xrange(20,90,2):
#	ax.view_init(elev=angle, azim=-60)
for angle in xrange(30,90,2):
	ax.view_init(elev=angle, azim=-120)
	plt.draw();
	plt.pause(.1)
	#plt.savefig((foldername+'Reconstruction-BeforeAfter_%d.png' % angle), bbox_inches="tight", dpi=300)

plt.show();

