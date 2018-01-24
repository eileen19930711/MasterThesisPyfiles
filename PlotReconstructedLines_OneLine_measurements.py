from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import NullFormatter
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

foldername = '/home/sheu_ch/la/A9/Analysis/Test2_img1line5/';
numseg=8
ii=4

## Read Statistics
totalimgnum = np.empty((0),int);
redundancy = np.empty((0),int);
sigma0hat = np.empty((0),float);
emean = np.empty((0),float);

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
			sigma0hat = np.append(sigma0hat,float(data[1]));
		elif 'e_mean' in line:
			data = line.split()
			emean = np.append(emean,float(data[1]));

fid.close()
num = totalimgnum.shape[0]

x = np.arange(1,num+1)



####### figure2: Residuals

f, (ax1,ax2) = plt.subplots(2, sharex=True)

ax1.plot(x, emean, '-', color='blueviolet',marker='o',markeredgewidth=0,markersize=4)

ax2.plot(x, sigma0hat, '-', label='residuals', color='blueviolet',marker='o',markeredgewidth=0,markersize=4)


ax1.set_xlim([np.min(x),np.max(x)])
ax2.set_xlim([np.min(x),np.max(x)])
ax2.set_ylim([np.min(sigma0hat),0.75])

ax1.set_ylabel('mean [pixel]')
ax2.set_ylabel('variance [pixel]')

ax2.set_xlabel('the i$^{th}$ segment')

ax2.get_xticklabels()[-1].set_visible(False)


ax2.legend(loc='lower left', bbox_to_anchor=(-0.05,-0.48),fontsize=9)


# make subplots close to each other
f.subplots_adjust(hspace=0)

plt.gcf().set_size_inches(6, 4)
plt.savefig((foldername+'Test_error.png'), bbox_inches="tight", dpi=300)


plt.clf()

####### figure3: Z test
Zvalue = emean / sigma0hat * np.sqrt(redundancy)
print(emean[0])
print(sigma0hat[0])
print(redundancy[0])
Z0975=1.96


fig1 = plt.figure(1)
ax = fig1.add_subplot(111);

x2 = np.arange(-4, 4, 0.01)
y2 = mlab.normpdf(x2, 0, 1)
plt.plot(x2, y2, 'b-', linewidth=1,label='normal probability density function')

plt.fill_between(x2, 0, y2, color='skyblue')

plt.fill_between(x2, 0, y2, where=y2<y2[204], facecolor='lightcoral')

for xxx in Zvalue:
	plt.axvline(x=xxx, color='k', linewidth=0.5)
plt.axvline(x=Zvalue[0], color='k', linewidth=0.5, label='$Z_{obs}$ of each segment')

plt.plot([-Z0975,-Z0975],[0,y2[204]], color='r')
plt.plot([Z0975,Z0975],[0,y2[204]], color='r')

plt.text(-Z0975-0.2, -0.05, '$-Z_{0.975}=-1.96$',color='r')
plt.text(Z0975-0.2, -0.05, '$Z_{0.975}=1.96$',color='r')

plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.ylim([0,0.45])

ax.legend(loc='lower left', bbox_to_anchor=(0,-0.45),fontsize=9)
plt.subplots_adjust(bottom=0.3)
plt.gcf().set_size_inches(8, 3)
plt.savefig((foldername+'Test_Ttest.png'), bbox_inches="tight", dpi=300)

plt.clf()




####### figure4: Residuals (the rejected ones are plotted in red)

Zreject = np.where(np.logical_or(Zvalue < -Z0975, Zvalue > Z0975))[0]
print(Zreject)



f, (ax1,ax2) = plt.subplots(2, sharex=True)

ax1.plot(x, emean, '-', color='blueviolet',marker='o',markeredgewidth=0,markersize=4)
ax1.plot(x[Zreject], emean[Zreject],'ro',markeredgewidth=0,markersize=6)

ax2.plot(x, sigma0hat, '-', label='residuals', color='blueviolet',marker='o',markeredgewidth=0,markersize=4)
ax2.plot(x[Zreject], sigma0hat[Zreject],'ro', label='population mean significantly non zero',markeredgewidth=0,markersize=6)


ax1.set_xlim([np.min(x),np.max(x)])
ax2.set_xlim([np.min(x),np.max(x)])
ax2.set_ylim([np.min(sigma0hat),0.75])

ax1.set_ylabel('mean [pixel]')
ax2.set_ylabel('variance [pixel]')

ax2.set_xlabel('the i$^{th}$ segment')

ax2.get_xticklabels()[-1].set_visible(False)


ax2.legend(loc='lower left', bbox_to_anchor=(-0.05,-0.48),fontsize=9)


# make subplots close to each other
f.subplots_adjust(hspace=0)

plt.gcf().set_size_inches(6, 4)
plt.savefig((foldername+'Test_error_rejected.png'), bbox_inches="tight", dpi=300)

