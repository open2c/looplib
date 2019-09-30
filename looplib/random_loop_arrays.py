import sys,os
import numpy as np


def uniform_loop_array(N, loop_size, loop_spacing):
    num_loops = int(N // (loop_size + loop_spacing))
    
    nonlooped_n = (
        N
        - num_loops * (loop_size + loop_spacing)
        - loop_spacing if num_loops else 0)
    shift = nonlooped_n // 2 if nonlooped_n > 0 else 0
    loops = [
        (shift + (loop_size + loop_spacing) * i,
         shift + (loop_size + loop_spacing) * i + loop_size)
        for i in range(num_loops)]

    return loops


def exponential_loop_array(
        N, loop_size, spacing, min_loop_size=3,
        loop_spacing_distr='uniform'):
    
    looplens = []
    spacers = []
    cumL = 0
    while True:
        looplens.append(min_loop_size
                        + np.round(np.random.exponential(loop_size-min_loop_size)))
        if loop_spacing_distr == 'exp':
            spacers.append(
                max(1, int(np.round(np.random.exponential(spacing-1)))))
        elif loop_spacing_distr == 'uniform':
            spacers.append(spacing)
        else: 
            raise ValueError('Unknown distribution of loop spacers')
            
        cumL += looplens[-1] + spacers[-1]
        if cumL > N-1:
            if len(looplens) > 1:
                looplens.pop(len(looplens) - 1)
                spacers.pop(len(spacers) - 1)
            break

    looplens, spacers = np.array(looplens), np.array(spacers)
    looplens = looplens * float(N - spacers.sum()) / (looplens.sum())

    loopstarts = np.r_[0, np.cumsum(looplens+spacers)[:-1]]
    loops = np.vstack([np.round(loopstarts), np.round(loopstarts + looplens)]).T
    loops = loops.astype('int')

    return loops


def two_layer_exponential_loops(N, outer_loop_size, outer_loop_spacing,
                                inner_loop_size, inner_loop_spacing,
                                offset=1):

    outer_loops = exponential_loop_array(
                    N, outer_loop_size, outer_loop_spacing,
                    min_loop_size=2*offset+inner_loop_spacing+1)

    if not inner_loop_size:
        return outer_loops

    inner_loops = []
    for l,r in outer_loops:
        looplen = r-l
        inner_loops += [(i[0] + l + offset, i[1] + l + offset)
                        for i in exponential_loop_array(looplen-2*offset,
                            inner_loop_size, inner_loop_spacing)]

    outer_loops, inner_loops  = np.array(outer_loops), np.array(inner_loops)
    return outer_loops, inner_loops


def gamma_loop_array(N, loop_size, loop_k, spacing=1, min_loop_size=3):
    looplens = []
    spacers = []
    cumL = 0
    if loop_size <= min_loop_size:
        return []
    while True:
        looplens.append(min_loop_size
            + np.round(np.random.gamma(loop_k, float(loop_size - min_loop_size)/loop_k)))
        spacers.append(spacing)
        cumL += looplens[-1] + spacers[-1]
        if cumL > N-1:
            if len(looplens) > 1:
                looplens.pop(len(looplens) - 1)
                spacers.pop(len(spacers) - 1)
            break

    looplens, spacers = np.array(looplens), np.array(spacers)
    looplens = looplens * float(N - 1 - spacers.sum()) / (looplens.sum())

    loopstarts = np.r_[0, np.cumsum(looplens+spacers)[:-1]]
    loops = np.vstack([np.round(loopstarts), np.round(loopstarts + looplens)]).T
    loops = loops.astype('int')

    return loops


def two_layer_gamma_loop_array(N,
                          outer_loop_size, outer_gamma_k, outer_loop_spacing,
                          inner_loop_size, inner_gamma_k, inner_loop_spacing,
                          outer_inner_offset=1):

    outer_loops = gamma_loop_array(
                    N,
                    outer_loop_size,
                    outer_gamma_k,
                    outer_loop_spacing,
                    2*outer_inner_offset+inner_loop_spacing+1)

    if not (inner_loop_size):
        return outer_loops

    inner_loops = []
    for l,r in outer_loops:
        looplen = r-l
        inner_loops += [(k[0] + l + outer_inner_offset, k[1] + l + outer_inner_offset)
                        for k in gamma_loop_array(
                            looplen-2*outer_inner_offset,
                            inner_loop_size,
                            inner_gamma_k,
                            inner_loop_spacing,
                            3)]

    outer_loops, inner_loops  = np.array(outer_loops), np.array(inner_loops)

    return outer_loops, inner_loops

