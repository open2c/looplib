import numpy as np
cimport numpy as np

cpdef get_parent_loops(l_sites, r_sites):
    """Take ndarrays with the positions of hierarchically nested loops
    and return the indices of their first parents. Returns -1 if a loop is not
    nested into other loops.
    """
    cdef np.int_t N = len(l_sites)
    cdef np.int_t[:] sortarg = np.argsort(np.r_[l_sites.astype(np.float64)+0.1,
                                                r_sites.astype(np.float64)-0.1])
    cdef np.int_t[:] looporder = np.take(np.r_[np.arange(1,N+1),-np.arange(1,N+1)],sortarg)
    cdef np.int_t[:] parents = -3 * np.ones(N, dtype=np.int)

    cdef np.int_t i = 0
    cdef np.int_t curloop = 0
    cdef np.int_t parent = 0
    cdef np.int_t[:] stack = -1 * np.ones(N+1, dtype=np.int)
    cdef np.int_t stack_level = 0
    for i in range(2*N):
        curloop = np.abs(looporder[i]) - 1
        is_bound = (l_sites[curloop] >= 0) * (r_sites[curloop] >= 0)
        is_left_leg = (looporder[i] > 0)
        if is_bound:
            if is_left_leg:
                parents[curloop] = stack[stack_level]
                stack_level += 1
                stack[stack_level] = curloop
            else:
                parent = stack[stack_level]
                stack_level -= 1
                assert curloop == parent, 'the loops are not nested'
        else:
            parents[curloop] = -2

    return np.array(parents)


cpdef list get_stationary_loops(
    np.int_t[:] prev_lsites,
    np.int_t[:] new_lsites,
    int delta):
    """
    Take two sorted arrays of loop positions and return the indices of those
    that did not move far (i.e. shifted by less than delta).
    """
    cdef np.int_t i = 0
    cdef np.int_t j = 0
    cdef list sync_boundaries = []
    while True:
        if abs(prev_lsites[i] - new_lsites[j]) <= delta:
            sync_boundaries.append((i,j))
            i += 1
            j += 1
        elif prev_lsites[i] > new_lsites[j]:
            j += 1
        else:
            i += 1
        if (i >= len(prev_lsites)) or (j >= len(new_lsites)):
            sync_boundaries.append((len(prev_lsites), len(new_lsites)))
            return sync_boundaries
