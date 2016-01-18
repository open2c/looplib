import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon

import looptools

def plot_interaction(l, r, n_lef=0, height_factor=1.0, max_height = 150, height=None, y=10):#, triag_width=100):
    arc_center = ((l+r)/2,y)
    arc_height = (min(max_height, (r-l)/2.0*height_factor)
                  if (height is None) else height)
    arc = Arc(xy=arc_center,
            width=r-l,
            height=arc_height,
            theta1=0,
            theta2=180,
            alpha=0.45,
            lw=5,
            color=(223.0/255.0,90/255.0,73/255.0),
            capstyle='round')
    plt.gca().add_artist(arc)

    if n_lef > 1 and arc_center[0] < plt.xlim()[1]:
        plt.text(x=arc_center[0],
                 y=arc_center[1]+arc_height/2+30,
                 horizontalalignment='center',
                 verticalalignment='center',
                 s=str(int(n_lef)),
                 fontsize=20)


def plot_lefs(l_sites, r_sites, L, site_width_bp = 600):
    plt.figure(figsize=(15,5))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(
        which='both',
        bottom='on',
        right='off',
        left='off',
        direction='out',
        top='off')
    plt.axhline(0, color='gray', lw=5,zorder=-1)
    plt.yticks([])
    plt.xticks([100*i*1000/float(site_width_bp) for i in range(16)],
               [100*i for i in range(16)],
               fontsize=20)
    plt.xlabel('chromosomal position, kb', fontsize=20)

    n_lefs = looptools.stack_lefs(l_sites,r_sites)
    for i in range(l_sites.size):
        plot_interaction(l_sites[i],r_sites[i],n_lefs[i])

    plt.xlim(-20,L+20)
    plt.ylim(-30,200)
