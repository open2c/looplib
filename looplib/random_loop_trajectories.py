import numpy as np

def exponential_min_loops(L, loop_size):
    initial_lefts = np.where(
        np.random.random(L / 2) < 1.0 / loop_size)[0] * 2
    initial_rights = initial_lefts + 1

    return zip(initial_lefts, initial_rights)


def gamma_min_loops(L, loop_size, shape):
    initial_lefts = [0]
    while True:
        initial_lefts.append(
            initial_lefts[-1]
            + int(np.round(
                (max(2,np.random.gamma(shape, loop_size/shape))))))
        if initial_lefts[-1] >= L:
            initial_lefts.pop(-1)
            break

    initial_lefts = np.array(initial_lefts)
    initial_rights = initial_lefts + 1
    return zip(initial_lefts, initial_rights)


def make_expanding_loops_trajectory(final_positions, move_prob=2.0, double_loops=False):
    
    initial_positions = [
        (i + (j-i) // 2, i + 1 + (j-i) // 2) for i, j in final_positions]

    trajectory = [initial_positions,]

    while True:
        new_position = []
        no_moves_allowed = True
        for i in range(len(trajectory[-1])):
            new_point = trajectory[-1][i]
            move_allowed = ((new_point[0] > final_positions[i][0])
                            and (new_point[1] < final_positions[i][1])) 
                            #(new_point != final_positions[i])
            no_moves_allowed = no_moves_allowed and not move_allowed

            if move_allowed and np.random.random() <= move_prob:
                new_point = (new_point[0] - 1, new_point[1] + 1)
            new_position.append(new_point)

        if no_moves_allowed:
            break

        trajectory.append(new_position)

    trajectory.append(final_positions)

    return trajectory

def make_random_loop_trajectory(initial_positions, L, move_prob=1.0,
                                double_loops=False, keep_identical_steps=True):

    assert all([i[1]>i[0] for i in initial_positions])
    assert all([j[0]>i[1] for (i,j) in zip(initial_positions[:-1],
                                           initial_positions[1:])])

    num_loops = len(initial_positions)
    trajectory = [initial_positions,]

    while True:
        new_positions = []
        prev_positions = trajectory[-1]
        no_moves_allowed = True
        for i in range(num_loops):
            new_left = max(prev_positions[i][0] - 1,
                           (0 if i==0 else new_positions[i-1][1] + 1))
            new_right = min(prev_positions[i][1] + 1,
                            (L-1 if i==num_loops-1 else prev_positions[i+1][0] - 1))
            new_point = (new_left, new_right)
            move_allowed = (new_point != prev_positions[i])
            no_moves_allowed = no_moves_allowed and not move_allowed

            if move_allowed and np.random.random() <= move_prob:
                new_positions.append(new_point)
            else:
                new_positions.append(prev_positions[i])

        if no_moves_allowed:
            break

        if (keep_identical_steps
            or not np.all(np.array([i[0] for i in prev_positions]) ==
                          np.array([i[0] for i in new_positions]))
            or not np.all(np.array([i[1] for i in prev_positions]) ==
                          np.array([i[1] for i in new_positions]))):
            trajectory.append(new_positions)

    return trajectory


def generate_cohesin_traj(ncoh, l_single_chain, l_sites, r_sites):
    """
    Generates the trajectory of cohesins along sister chromatids, assuming that
    they can be freely pushed along the chromatin by condensins.
    """

    tmax,ncond = l_sites.shape
    coh_pos = -1*np.ones((tmax,ncoh*2),dtype=np.int)

    left_coh_pos = np.sort(np.random.randint(0, l_single_chain+1, ncoh))
    coh_pos[0] = np.hstack([left_coh_pos, l_single_chain + left_coh_pos])

    smc_hopped = (np.diff(l_sites, axis=0) > 0) + (np.diff(r_sites, axis=0) < 0)
    for t in xrange(1,tmax):
        new_pos = np.arange(2*l_single_chain+1)
        for i in range(ncond):
            if not smc_hopped[t-1][i]:
                new_pos[r_sites[t-1,i]+1:r_sites[t,i]+1] = r_sites[t,i]+1
                new_pos[l_sites[t,i]:l_sites[t-1,i]+1] = l_sites[t,i]
        coh_pos[t]=new_pos[coh_pos[t-1]]
    coh_pos = np.dstack([coh_pos[:,:ncoh], coh_pos[:,ncoh:]])

    # remove artifact cis-links
    coh_pos[:,:,0][coh_pos[:,:,0] == l_single_chain] -= 1
    coh_pos[:,:,1][coh_pos[:,:,1] == 2 * l_single_chain] -= 1

    return coh_pos