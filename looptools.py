import numpy as np

import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True)
from looptools_c import get_parent_loops, get_stationary_loops

def convert_loops_to_sites(loops, r_sites=None):
    """
    Convert a list of loops defined by tuples (left_site, right_side) into
    two np.arrays left_sites, right_sites.
    """
    if r_sites is None:
        if (issubclass(type(loops), list)
            and issubclass(type(loops[0]), tuple)):
            l_sites, r_sites = np.array([i for i,j in loops]), np.array([j for i,j in loops])
        elif (issubclass(type(loops), np.ndarray) and (loops.ndim == 2)):
            if (loops.shape[0] == 2):
                return loops[0], loops[1]
            elif (loops.shape[1] == 2):
                return loops[:,0], loops[:,1]
            else:
                raise Exception('Unknown format of loop array')
        else:
            raise Exception('Unknown format of loop array')
    else:
        assert ((issubclass(type(loops), list)
                or issubclass(type(loops), np.ndarray))
                and (issubclass(type(r_sites), list)
                     or issubclass(type(r_sites), np.ndarray)))
        l_sites = np.array(loops, copy=False)
        r_sites = np.array(r_sites, copy=False)

    return l_sites, r_sites

def get_roots(l_sites, r_sites=None):
    """Return the indices of root loops (i.e. loops not nested into
    other loops).
    """

    l_sites, r_sites = convert_loops_to_sites(l_sites, r_sites)

    try:
        parent_loops = get_parent_loops(l_sites, r_sites)
    except Exception as e:
        print 'Cannot find root loops: ', e.message
        return []

    return (parent_loops == -1)

def get_root_births_deaths(prev_lsites, new_lsites, delta):
    """
    Estimate the number of root loops that divided or died between the two time
    frames. This number is estimated by finding groups of root loops with
    stationary outer borders and measuring the change in the number of roots
    within these groups.
    """
    sync_boundaries = get_stationary_loops(
        np.sort(prev_lsites), np.sort(new_lsites), delta)
    births, deaths = 0,0
    for i,j in zip(sync_boundaries[1:], sync_boundaries[:-1]):
        n_loop_change = i[0] - j[0] - i[1] + j[1]
        births += max(0, n_loop_change)
        deaths += max(0, -n_loop_change)

    return births, deaths

def avg_num_branching_points(parents):
    sameparent = (parents[:, None] == parents[None, :])
    np.fill_diagonal(sameparent, False)
    numsisters = (sameparent.sum(axis=1)).astype('float')
    numsisters[parents == -1] = 0 # root loops don't have sisters
    # normalize to the number of sisters
    numsisters[numsisters != -1] /= (numsisters[numsisters != -1] + 1.0)
    numbranches = numsisters.sum()
    # normalize to the number of roots
    avgnumbranches = numbranches / float((parents == -1).sum())
    return avgnumbranches

def get_loop_branches(parents, l_sites=None):
    '''Get the list of list of daughter loops. If `l_sites` is provided,
    sort the daughter loops according to their position along the loop.
    '''
    nloops = len(parents)
    daughters = [np.where(parents==i)[0] for i in range(nloops)]
    if not (l_sites is None):
        for i in range(nloops):
            daughters[i] = daughters[i][np.argsort(l_sites[daughters[i]])]
    return daughters

def stack_lems(l_sites, r_sites):
    order = np.argsort(l_sites)
    n_conds = np.ones(l_sites.size)
    parent_i, i = 0,0
    while True:
        if ((l_sites[order[i+1]]==l_sites[order[i]]+1)
            and (r_sites[order[i+1]]==r_sites[order[i]]-1)):
            n_conds[order[i+1]] -= 1
            n_conds[order[parent_i]] += 1
        else:
            parent_i = i+1
        i += 1
        if i >= l_sites.size-1:
            break
    return n_conds

#def plot_loop(l, r, n_cond=0, height=None, y=5, label_shift=0, fontsize=14):
#from matplotlib.patches import Arc
#    arc_center = ((l+r)/2,y)
#    arc_height = ((r-l) if (height is None) else height)
#    e = Arc(xy=arc_center,
#            width=r-l,
#            height=arc_height,
#            theta1=0,
#            theta2=180,
#            alpha=0.3,
#            lw=5)
#    plt.gca().add_artist(e)
#    if n_cond > 1:
#        plt.text(x=arc_center[0],
#                 y=arc_center[1]+arc_height*0.5 + label_shift,
#                 horizontalalignment='center',
#                 verticalalignment='center',
#                 s=str(int(n_cond)),
#                 fontsize=fontsize)

#def plot_lems(l_sites, r_sites, L, fontsize=14):
#    plt.figure(figsize=(15,5))
#    plt.gca().spines['top'].set_visible(False)
#    plt.gca().spines['left'].set_visible(False)
#    plt.gca().spines['right'].set_visible(False)
#    plt.gca().tick_params(
#        which='both',
#        bottom='on',
#        right='off',
#        left='off',
#        direction='out',
#        top='off')
#    plt.yticks([])
#
#    n_conds = stack_lems(l_sites,r_sites)
#    max_l = max([i-j for (i,j) in zip(r_sites, l_sites)])
#    for i in range(l_sites.size):
#        plot_loop(l_sites[i],r_sites[i], n_conds[i],
#                  label_shift=max_l * 0.05,
#                  fontsize=fontsize)
#
#    plt.xlim(-L*0.025,L*1.025)
#    plt.ylim(0, max_l * 0.7)


def get_backbone(l_sites, r_sites, rootsMask=None, N=None, include_tails=True):
    backboneidxs = []
    if rootsMask is None:
        rootsMask = get_roots(l_sites, r_sites)

    rootsSorted = np.where(rootsMask)[0][np.argsort(l_sites[rootsMask])]

    if include_tails and (N is None):
        raise Exception('If you want to include tails, please specify the length of the chain')

    for i in range(len(rootsSorted)-1):
        backboneidxs.append(
            np.arange(r_sites[rootsSorted[i]], l_sites[rootsSorted[i+1]]+1, dtype=np.int))

    if include_tails:
        backboneidxs.insert(0,
            np.arange(0, l_sites[rootsSorted[0]] + 1, dtype=np.int))
        backboneidxs.append(
            np.arange(r_sites[rootsSorted[-1]], N, dtype=np.int))

    backboneidxs = np.concatenate(backboneidxs)
    return backboneidxs

