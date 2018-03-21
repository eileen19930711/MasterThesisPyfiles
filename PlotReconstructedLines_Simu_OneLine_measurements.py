from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import NullFormatter
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from matplotlib.ticker import MaxNLocator

#foldername = 'D:/4-5th semester -MasterThesisDLR/files/Analysis/Simu_0306/';
foldername = '/home/sheu_ch/la/A9/Analysis/Simu_0306/';
ii=0

## Read Statistics
totalimgnum = np.empty((0),int);
redundancy = np.empty((0),int);
sigma0hat = np.empty((0),float);

with open((foldername+'statistics_'+str(ii)+'.txt'),'r') as fid:
	for line in fid:
		if 'totalimgnum' in line:
			data = line.split();
			totalimgnum = np.append(totalimgnum, int(data[1]))

		elif 'redundancy' in line:
			data = line.split()
			redundancy = np.append(redundancy,int(data[1]))

		elif 'sigma0hat' in line:
			data = line.split();
			sigma0hat = np.append(sigma0hat,float(data[1]));

fid.close()
num = totalimgnum.shape[0]

x = np.arange(1,num+1)

####### Read Known Errors in Col Row -all

crall = np.empty((0),float);
try:
	cr = np.loadtxt((foldername+'randomerrors_cr.txt'), usecols=(0, 1), unpack=True)
	crall = np.concatenate((crall,cr[0,0:].T),axis=0)
	crall = np.concatenate((crall,cr[1,0:].T),axis=0)
except IOError as e:
	print('file randomerrors_cr.txt not found')



####### Read Known Errors in Col Row -each LS

re_mean = np.empty((0),float);
re_std = np.empty((0),float);

for i in range(1,num+1): #,1959)
	try:
		rcr = np.loadtxt((foldername + 'randomerrors_cr_'+str(i)+'.txt'), usecols=(0, 1), unpack=True)
		re_mean = np.append(re_mean,float(np.mean(rcr)))
		re_std = np.append(re_std,float(np.std(rcr)))

	except IOError as e:
		print('file randomerrors_cr_'+str(i)+'.txt not found')




####### Read Residuals -each LS

e_mean = np.empty((0),float);
e_std = np.empty((0),float);

for i in range(1,num+1): #,1959)
	try:
		ecr = np.loadtxt((foldername + 'ehat_0_' + str(i) + '.txt'), usecols=(0, 1), unpack=True)
		e_mean = np.append(e_mean,float(np.mean(ecr[0,0:]+ecr[1,0:])))
		e_std = np.append(e_std,float(np.std(ecr[0,0:]+ecr[1,0:])))
		
	except IOError as e:
		print('ehat_0_'+str(i)+'.txt not found')




####### figure1: Random Errors Histogram

f, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(crall, 50, normed=1, facecolor='mediumaquamarine', alpha=1,label='the added errors in image coordinates')
# add a 'best fit' line
y = mlab.normpdf( bins, np.mean(crall), np.std(crall))
ax.plot(bins, y, 'b-', linewidth=1,label='normal probability density function')

ax.text(np.mean(crall), y.max(axis=0)+0.05, '$\mathcal{N}(%.3f,%.3f^2)$' %(np.mean(crall),np.std(crall)),color='blue')

#
ax.set_xlabel('the added errors [pixel]')
ax.set_ylabel('probability density')
ax.set_ylim([0,1])

ax.legend(loc='lower left', bbox_to_anchor=(-0.05,-0.60),fontsize=9)

plt.gcf().set_size_inches(6, 2)
plt.savefig((foldername+'Simu_errorhist_1.png'), bbox_inches="tight", dpi=300)

plt.clf()


####### figure2: Residuals

f, (ax1,ax2) = plt.subplots(2, sharex=True)

ax1.plot(x, re_mean, '-', color='deeppink',marker='o',markeredgewidth=0,markersize=4)
ax1.plot(x, e_mean, '-', color='blueviolet',marker='o',markeredgewidth=0,markersize=4)

ax2.plot(x, re_std, '-', label='the added errors', color='deeppink',marker='o',markeredgewidth=0,markersize=4)
ax2.plot(x, e_std, '-', label='residuals', color='blueviolet',marker='o',markeredgewidth=0,markersize=4)



#maxmean = np.max([re_mean,e_mean])
#minmean = np.min([re_mean,e_mean])

#maxstd = np.max([re_std,e_std])
#minstd = np.min([re_std,e_std])

#ax1.set_ylim([minmean,maxmean])
#ax2.set_ylim([minstd,maxstd])

ax1.set_xlim([np.min(x),np.max(x)])
ax2.set_xlim([np.min(x),np.max(x)])

ax1.set_ylabel('mean [pixel]')
ax2.set_ylabel('standard deviation [pixel]')

ax2.set_xlabel('the i$^{th}$ node')

ax2.get_xticklabels()[-1].set_visible(False)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

ax2.legend(loc='lower left', bbox_to_anchor=(-0.05,-0.48),fontsize=9)


# make subplots close to each other
f.subplots_adjust(hspace=0)

plt.gcf().set_size_inches(6, 4)
plt.savefig((foldername+'Simu_error_1.png'), bbox_inches="tight", dpi=300)


plt.clf()

####### figure3: Z test

Tvalue = (e_mean - re_mean) / np.sqrt(e_std ** 2 + re_std ** 2) * np.sqrt(redundancy)
T0975=1.968


fig1 = plt.figure(1)
ax = fig1.add_subplot(111);

x = np.arange(-4, 4, 0.01)
y = mlab.normpdf(x, 0, 1)
plt.plot(x, y, 'b-', linewidth=1,label='normal probability density function')

plt.fill_between(x, 0, y, color='skyblue')

plt.fill_between(x, 0, y, where=y<y[204], facecolor='lightcoral')

for xxx in Tvalue:
	plt.axvline(x=xxx, color='k', linewidth=0.5)
plt.axvline(x=Tvalue[0], color='k', linewidth=0.5, label='$T_{obs}$ of each segment')

plt.plot([-T0975,-T0975],[0,y[204]], color='r')
plt.plot([T0975,T0975],[0,y[204]], color='r')

plt.text(-T0975-0.2, -0.05, '$-T_{0.975}=-1.968$',color='r')
plt.text(T0975-0.2, -0.05, '$T_{0.975}=1.968$',color='r')

plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.ylim([0,0.45])

ax.legend(loc='lower left', bbox_to_anchor=(0,-0.45),fontsize=9)
plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(8, 3)
plt.savefig((foldername+'Simu_Ttest.png'), bbox_inches="tight", dpi=300)

