import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as mtick

params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize': 15,}
pylab.rcParams.update(params)

name = 'all-test-2'
param = 'optimizer+lr'

df = pd.read_csv(f'results/{name}.csv', index_col=0)
df = df.loc[df['param'] == param]
df['ACC'] = pd.Series([round(val * 100, 2) for val in df['ACC']], index = df.index)

""" #df = pd.DataFrame({'loss': data['LOSS'].values, 'accuracy': data['ACC'].values}, index=data['embeding'].values)
df = pd.DataFrame({'accuracy': df['ACC'].values}, index=df[param].values)
ax = df.plot.barh(figsize=(8,4), title='Chart', xlabel='Accuracy %', ylabel='Embeding')
#ax.bar_label(ax, padding=3, fmt='%.1f%%')
ax.bar_label(ax.containers[0], label_type='edge', padding=5, fmt='%.1f%%')
ax.set_xlim(0, 105)
plt.tight_layout()
#ax.bar_label(ax)
plt.legend(loc='upper right', fontsize=10)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
ax.get_legend().remove()
#ax.set_xlabel('xlabel', fontsize=10)
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
plt.show() """


sgd = df.loc[df['optimizer'] == 'SGD']['ACC'].values
adam = df.loc[df['optimizer'] == 'Adam']['ACC'].values
nadam = df.loc[df['optimizer'] == 'Nadam']['ACC'].values
x = df.loc[df['optimizer'] == 'Nadam']['ACC'].values
index = ['lr 0.01', 'lr 0.05', 'lr 0.001', 'lr 0.005', 'lr 0.0001', 'lr 0.0005']
df = pd.DataFrame({'SGD': sgd, 'Adam': adam, 'Nadam': nadam, 'X': x}, index=index)
ax = df.plot.bar(figsize=(8,4), rot=0, width=0.8)
for i in range(len(ax.containers)):
    #max = max(c._height for c in ax.containers[i])
    ax.bar_label(ax.containers[i], label_type='edge', padding=5, fmt='%.1f', size=8)
#print(max(c._height for c in ax.containers[3]))
#print(ax.containers[1][2]._height)
ax.set_ylim(0, 105)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
ax.legend(loc='lower left',  ncol=3)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
plt.show()



""" species = ('0.01', '0.05', '0.001', '0.005', '0.0001', '0.0005')
penguin_means = {
    'SGD': (18.35, 8.43, 4.98),
    'Adam': (3.89, 48.83, 4.750),
    'Nadam': (18.995, 95.82, 2.1719),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fmt='%.1f%%')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left')
ax.set_ylim(0, 105)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
plt.show()
 """
