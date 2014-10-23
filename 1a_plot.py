from pylab import *
import pandas as pd


df =pd.read_csv("data/PS04_1a.dat", names=range(1,35))
p1 = pd.DataFrame({'max': df.max(), 'mean':df.mean()})
ax = p1.plot(marker='.', markersize=10, title='')
ax.set_xlabel("The number of rounds, t")
ax.set_ylabel("Log-likelihood")
fig = ax.get_figure()
fig.savefig('ps4-1a-fig1.pdf')
