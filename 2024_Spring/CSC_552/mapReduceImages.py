import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

time = [0,28,35,44,52,65,73,80,110]
inpt = [0,3,9,25,32,27,0.3,0.01,0]
fs = 29

fig,ax = plt.subplots(figsize=(8,3.5))
ax.plot(time,inpt,lw=4,c="#0070c0")
ax.set_xlabel("seconds",fontsize=fs)
ax.set_ylabel("Input (GB/s)",fontsize=fs,labelpad=10)
ax.margins(x=0.01)
for tick in (ax.get_xticklabels() + ax.get_yticklabels()):
    tick.set_fontsize(fs-5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("performance.png",bbox_inches="tight",dpi=200)
plt.close()