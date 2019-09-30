from __future__ import division, print_function
import numpy as np
import collections

#import pyximport; pyximport.install(
#    setup_args={"include_dirs":np.get_include()},
#    reload_support=True)
from .looptools_c import get_parent_loops, get_stationary_loops

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
        print('Cannot find root loops: ', e.message)
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
    children = [np.where(parents==i)[0] for i in range(nloops)]
    if not (l_sites is None):
        for i in range(nloops):
            children[i] = children[i][np.argsort(l_sites[children[i]])]
    return children

def stack_lefs(l_sites, r_sites):
    """Identify groups of stacked LEFs (i.e. tightly nested LEFs)
    """
    order = np.argsort(l_sites)
    n_lefs = np.ones(l_sites.size)
    parent_i, i = 0,0
    while True:
        if ((l_sites[order[i+1]]==l_sites[order[i]]+1)
            and (r_sites[order[i+1]]==r_sites[order[i]]-1)):
            n_lefs[order[i+1]] -= 1
            n_lefs[order[parent_i]] += 1
        else:
            parent_i = i+1
        i += 1
        if i >= l_sites.size-1:
            break
    return n_lefs

def get_backbone(l_sites, r_sites=None, rootsMask=None, N=None, include_tails=True):
    """Find the positions between the root loops aka the backbone.
    """
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

def get_n_leafs(idx, children):
    if isinstance(idx, collections.Iterable):
        return np.array([get_n_leafs(i, children) for i in idx])
    else:
        if len(children[idx])==0:
            return 1
        else:
            return sum([get_n_leafs(child, children) 
                        for child in children[idx]])
