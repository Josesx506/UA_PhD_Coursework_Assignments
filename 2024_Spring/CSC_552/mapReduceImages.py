import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams["font.family"] = "Arial"

def image_1():
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig("performance.png",bbox_inches="tight",dpi=200,transparent=True)
    plt.close()


def image_2():
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    for i in range(10):
        ax.add_patch(plt.Rectangle((-i*0.2, -i*0.2), 7.85, 7.85,fc="w",ls="-", ec="k"))
        ax.set_xlim(-2,8.0),ax.set_ylim(-2,8.0)
    text =str(random.getstate()).replace("(","-").replace(")","-").split("-")[2].split(",")
    text = [ascii(chr(int(val.strip()[:6]))) for val in text]
    fit_text = []
    for j in range(46):
        fit_text.append("  ".join(text[j * 6:j * 6 + 6]))
    fit_text = "\n".join(fit_text)
    ax.text(-1.7,-1.7,fit_text,fontsize=5.4,family=["Herculanum"])
    ax.axis("off")
    plt.savefig("bytes_search.png",bbox_inches="tight",dpi=200,transparent=True)
    plt.close()

# image_2()