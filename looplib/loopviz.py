from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon

from . import looptools

def plot_interaction(
    l, r, 
    label=None,
    height_factor=1.0,
    max_height=100,
    height=None,
    y=0,
    plot_text=True,
    color='tomato',
    alpha=0.45,
    lw=5):
    """Visualize an individual loop with an arc diagram.

    Parameters:
    l (float): The left position of the loop.
    r (float): The right position of the loop.
    label (str, optional): A label to annotate the loop.
    height_factor (float, optional): The factor to determine the height of the arc. Defaults to 1.0.
    max_height (int, optional): The maximum height of the arc. Defaults to 100.
    height (float, optional): The specific height of the arc. Defaults to None.
    y (float, optional): The y-coordinate of the arc center. Defaults to 0.
    plot_text (bool, optional): Whether to plot the text label. Defaults to True.
    color (str, optional): The color of the arc. Defaults to 'tomato'.
    alpha (float, optional): The transparency of the arc. Defaults to 0.45.
    lw (int, optional): The line width of the arc. Defaults to 5.
    """
    arc_center = ((l+r)/2,y)
    arc_height = (min(max_height, (r-l)/2.0*height_factor)
                  if (height is None) else height)
    arc = Arc(xy=arc_center,
            width=r-l,
            height=arc_height,
            theta1=0,
            theta2=180,
            alpha=alpha,
            lw=lw,
            color=color,
            capstyle='round')
    plt.gca().add_artist(arc)

    if label and arc_center[0] < plt.xlim()[1] and plot_text:
        plt.text(x=arc_center[0],
                 y=arc_center[1]+arc_height/2+20,
                 horizontalalignment='center',
                 verticalalignment='center',
                 s=label,
                 fontsize=20,
                 color=color
                )

def prepare_canvas(
    L,
    site_width_bp=None,
    **kwargs
):
    """
    Prepare the canvas for loop visualization.

    Parameters:
    - L (int): The length of the canvas.
    - site_width_bp (int, optional): The width of the site in base pairs. Defaults to None.
    - **kwargs: Additional keyword arguments for customizing the canvas.

    Returns:
    None
    """

    plt.figure(figsize=kwargs.get('figsize', (15, 5)))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(
        which='both',
        bottom=True,
        right=False,
        left=False,
        top=False,
        direction='out')
    plt.xlim(-20, L + 20)
    plt.ylim(-30, 120)

    plt.yticks([])

    plt.axhline(-10, color='gray', lw=5, zorder=-1)

    if site_width_bp:
        set_ticks(L, site_width_bp)
        
def set_ticks(L, site_width_bp=200, tick_spacing_bp=1e6,):
    tick_units = (1e6, "Mb") if tick_spacing_bp >= 1e6 else (1e3, "kb")
    plt.xticks([i/site_width_bp for i in np.arange(0,L*site_width_bp+1,tick_spacing_bp)],
               [f'{i//tick_units[0]:.0f}' for i in np.arange(0,L*site_width_bp+1,tick_spacing_bp)],
               fontsize=20)
    plt.xlabel(f'chromosomal position, {tick_units[1]}', fontsize=20)
    
    
def plot_lefs(
        l_sites, 
        r_sites, 
        colors='tomato',
        **kwargs):
    """Plot an arc diagram for a list of loops.

    Parameters:
    l_sites (ndarray): Array of left sites.
    r_sites (ndarray): Array of right sites.
    colors (str or list or tuple or ndarray, optional): Color(s) for the arcs. Defaults to 'tomato'.
    **kwargs: Additional keyword arguments to be passed to the plot_interaction function.

    """
    
    n_lefs = looptools.stack_lefs(np.vstack([l_sites,r_sites]).T)
    for i in range(l_sites.size):
        plot_interaction(
            l_sites[i],
            r_sites[i],
            n_lefs[i],
            color=colors[i] if (type(colors) in (list, tuple, np.ndarray)) else colors,
            **kwargs)


