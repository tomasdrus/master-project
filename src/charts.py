import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
         'axes.labelsize': 14,
         'axes.titlesize': 16,}
pylab.rcParams.update(params)

name = 'results-all'

data = pd.read_csv(f'results/{name}.csv', index_col=0)
data = data.loc[data['param'] == 'fmax']

#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot
df = pd.DataFrame({'accuracy': data['ACC'].values,'loss': data['LOSS'].values}, index=data['batch'].values)
ax = df.plot.barh(figsize=(8,5), title='Chart', xlabel='sfafa', ylabel='sfafa')
plt.legend(loc='upper right', fontsize=20)
ax.get_legend().remove()
#ax.set_xlabel('xlabel', fontsize=10)
plt.show()

