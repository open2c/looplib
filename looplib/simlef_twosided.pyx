###cython: profile=True
##cython: boundscheck=False
##cython: wraparound=False
##cython: nonecheck=False
##cython: initializedcheck=False
from __future__ import division, print_function
cimport cython
import numpy as np
cimport numpy as np

#mport numpy as np
from cpython cimport bool
from heapq import heappush, heappop

from libc.stdlib cimport rand, srand, RAND_MAX
srand(0)
np.random.seed()

cdef inline np.int64_t rand_int(int N_MAX):
    return np.random.randint(N_MAX)
    #return rand() % N_MAX

cdef inline np.float64_t rand_exp(np.float64_t mean):
    return np.random.exponential(mean)

cdef inline float rand_float():
    #return np.random.random()
    return rand() / float(RAND_MAX)

cdef inline int64sign(np.int64_t x):
    if x > 0:
        return +1
    else:
        return 0

cdef inline int64abs(np.int64_t x):
    if x > 0:
        return x
    else:
        return -x

cdef inline int64not(np.int64_t x):
    if x == 0:
        return 1
    else:
        return 0

cdef class State:
    cdef np.int64_t L
    cdef np.int64_t N
    cdef np.int64_t [:] lattice
    cdef np.int64_t [:] l_extendable, r_extendable
    cdef np.int64_t [:] l_shrinkable, r_shrinkable
    cdef np.int64_t [:] l_sites, r_sites
    cdef list recently_updated

    def __cinit__(self, L, N, init_l_sites=None, init_r_sites=None):
        self.L = L
        self.N = N
        self.lattice = np.zeros(L, dtype=np.int64)
        self.l_extendable = np.zeros(N, dtype=np.int64)
        self.r_extendable = np.zeros(N, dtype=np.int64)
        self.l_shrinkable = np.zeros(N, dtype=np.int64)
        self.r_shrinkable = np.zeros(N, dtype=np.int64)
        self.l_sites = -1 * np.ones(N, dtype=np.int64)
        self.r_sites = -1 * np.ones(N, dtype=np.int64)
        self.recently_updated = list()

        cdef np.int64_t i

        # Initialize non-random loops
        for i in range(self.N):
            # Check if the loop is preinitialized.
            if (init_l_sites[i] < 0) or (init_r_sites[i] < 0):
                continue

            # Check if the preinitialized values make sense.
            assert self.lattice[init_l_sites[i]] == 0
            assert self.lattice[init_r_sites[i]] == 0
            assert init_r_sites[i] > init_l_sites[i]

            # Populate a site.
            self.l_sites[i] = init_l_sites[i]
            self.r_sites[i] = init_r_sites[i]
            self.lattice[self.l_sites[i]] = - i - 1
            self.lattice[self.r_sites[i]] = i + 1
            self.update_extendability(self.l_sites[i])
            self.update_extendability(self.r_sites[i])

        ## Initialize random loops.
        #cdef np.int64_t site
        #for i in range(self.N):
        #    # Check if the loop is not preinitialized.
        #    if (self.l_sites[i] >= 0) and (self.r_sites[i] >= 0):
        #        continue

        #    # Find a site.
        #    while True:
        #        site = rand_int(self.L - 1)
        #        if (self.lattice[site] == 0 and self.lattice[site+1] == 0):
        #           break

        #    # Populate a site.
        #    self.l_sites[i] = site
        #    self.r_sites[i] = site + 1
        #    self.lattice[self.l_sites[i]] = - i - 1
        #    self.lattice[self.r_sites[i]] = i + 1
        #    self.update_extendability(self.l_sites[i])
        #    self.update_extendability(self.r_sites[i])

    cdef np.int64_t update_extendability(State self, np.int64_t pos) except 0:
        """
        """

        cdef np.int64_t this_leg, prev_leg, next_leg
        this_leg = self.lattice[pos]
        prev_leg = self.lattice[pos-1] if pos > 0 else 0
        next_leg = self.lattice[pos+1] if pos < self.L - 1 else 0

        if this_leg != 0:
            self.recently_updated.append(this_leg - 1 if this_leg > 0 else - this_leg - 1)
        if next_leg != 0:
            self.recently_updated.append(next_leg - 1 if next_leg > 0 else - next_leg - 1)
        if prev_leg != 0:
            self.recently_updated.append(prev_leg - 1 if prev_leg > 0 else - prev_leg - 1)

        assert (pos >= 0) and (pos < self.L)
        assert (this_leg >= - self.N - 1) and (this_leg <= self.N + 1)
        assert (prev_leg >= - self.N - 1) and (prev_leg <= self.N + 1)
        assert (next_leg >= - self.N - 1) and (next_leg <= self.N + 1)

        cdef np.int64_t new_extendability = (this_leg == 0)
        # Update extendability of the leg on the left.
        if prev_leg > 0:
            self.r_extendable[prev_leg - 1] = new_extendability
        elif prev_leg < 0:
            self.l_shrinkable[- prev_leg - 1] = new_extendability

        # Update extendability of the leg on the right.
        if next_leg > 0:
            self.r_shrinkable[next_leg - 1] = new_extendability
        elif next_leg < 0:
            self.l_extendable[- next_leg - 1] = new_extendability

        # Update extendability at the current position.
        if this_leg > 0:
            self.r_extendable[this_leg - 1] = 1 if (next_leg == 0) and (pos < self.L-1) else 0
            self.r_shrinkable[this_leg - 1] = 1 if (prev_leg == 0) else 0
        elif this_leg < 0:
            self.l_extendable[- this_leg - 1] = 1 if (prev_leg == 0) and (pos > 0) else 0
            self.l_shrinkable[- this_leg - 1] = 1 if (next_leg == 0) else 0

        return 1

    cdef reset_recently_updated(State self):
        del self.recently_updated[:]

    cdef np.int64_t move_leg(State self, np.int64_t loop_idx,
            np.int64_t is_right, np.int64_t new_pos) except 0:
        assert (self.lattice[new_pos] == 0)
        cdef np.int64_t prev_pos

        if is_right > 0:
            prev_pos = self.r_sites[loop_idx]
            self.r_sites[loop_idx] = new_pos
        else:
            prev_pos = self.l_sites[loop_idx]
            self.l_sites[loop_idx] = new_pos

        if prev_pos >= 0:
            assert (self.lattice[prev_pos] != 0)
            self.lattice[prev_pos] = 0
        self.lattice[new_pos] = is_right * (loop_idx + 1)

        ## Reset the extendabilities of the sites.
        cdef np.int64_t status = 1
        if prev_pos >= 0:
            status *= self.update_extendability(prev_pos)
        status *= self.update_extendability(new_pos)
        assert status

        return 1

    cdef np.int64_t extend_leg(State self, np.int64_t loop_idx, np.int64_t is_right):
        """
        The variable `is_right` can only take values +1 or -1.
        """

        cdef np.int64_t new_pos = self.r_sites[loop_idx] if is_right > 0 else self.l_sites[loop_idx]
        new_pos += is_right
        return self.move_leg(loop_idx, is_right, new_pos)

    cdef np.int64_t shrink_leg(State self, np.int64_t loop_idx, np.int64_t is_right):
        """
        The variable `is_right` can only take values +1 or -1.
        """

        cdef np.int64_t new_pos = self.r_sites[loop_idx] if is_right > 0 else self.l_sites[loop_idx]
        new_pos -= is_right
        return self.move_leg(loop_idx, is_right, new_pos)

    cdef np.int64_t check_state(State self):
        okay = 1
        cdef np.int64_t i
        for i in range(self.N):
            if (self.l_sites[i] == self.r_sites[i]):
                print(self.l_sites[i], self.r_sites[i])
                okay = 0
            elif (self.l_sites[i] < 0) or (self.l_sites[i] >= self.L):
                print(self.l_sites[i])
                okay = 0
            elif (self.r_sites[i] < 0) or (self.r_sites[i] >= self.L):
                print(self.r_sites[i])
                okay = 0
            elif i > 0 and self.l_extendable[i] and self.lattice[self.l_sites[i]-1] != 0:
                print(i, 'should not be l-extandable')
            elif i > 0 and self.r_shrinkable[i] and self.lattice[self.r_sites[i]-1] != 0:
                print (i, 'should not be r-shrinkable')
            elif i < self.L-1 and self.r_extendable[i] and self.lattice[self.r_sites[i]+1] != 0:
                print (i, 'should not be r-extandable')
            elif i < self.L-1 and self.l_shrinkable[i] and self.lattice[self.l_sites[i]+1] != 0:
                print (i, 'should not be l-shrinkable')
        return okay

cdef np.int64_t move_loop(State state, np.int64_t loop_idx) except 0:
    cdef np.int64_t site
    while True:
        site = rand_int(state.L-1)
        if (state.lattice[site] == 0 and state.lattice[site+1] == 0):
            break

    cdef np.int64_t status = 3
    status *= state.move_leg(loop_idx, -1, site)
    status *= state.move_leg(loop_idx, 1, site+1)

    return status

cdef np.int64_t extend_loop(State state, np.int64_t loop_idx, np.int64_t ONE_SIDE_EXTEND) except 0:
    cdef np.int64_t status = 1
    if (state.l_extendable[loop_idx] * state.r_extendable[loop_idx]):
        assert (state.lattice[state.r_sites[loop_idx] + 1] == 0)
        assert (state.lattice[state.l_sites[loop_idx] - 1] == 0)
        status *= state.extend_leg(loop_idx, 1)
        status *= state.extend_leg(loop_idx, -1)
        return status

    elif ONE_SIDE_EXTEND:
        if state.r_extendable[loop_idx]:
            assert (state.lattice[state.r_sites[loop_idx] + 1] == 0)
            status *= state.extend_leg(loop_idx, 1)
        elif state.l_extendable[loop_idx]:
            assert (state.lattice[state.l_sites[loop_idx] - 1] == 0)
            status *= state.extend_leg(loop_idx, -1)
        return status

    return 0

cdef np.int64_t shrink_loop(State state, np.int64_t loop_idx, np.int64_t ONE_SIDE_EXTEND) except 0:
    cdef np.int64_t status = 2
    if ONE_SIDE_EXTEND:
        if (state.r_sites[loop_idx] - state.l_sites[loop_idx] == 2):
            assert (state.lattice[state.l_sites[loop_idx] + 1] == 0)
            status *= state.shrink_leg(loop_idx, 1 if rand_float() < 0.5 else -1)
            return status
        elif state.r_shrinkable[loop_idx] or state.l_shrinkable[loop_idx]:
            if state.r_shrinkable[loop_idx]:
                status *= state.shrink_leg(loop_idx, 1)
            if state.l_shrinkable[loop_idx]:
                status *= state.shrink_leg(loop_idx, -1)
            return status

    elif (state.l_shrinkable[loop_idx] and state.r_shrinkable[loop_idx]):
        if (state.r_sites[loop_idx] - state.l_sites[loop_idx] > 2):
            status *= state.shrink_leg(loop_idx, 1)
            status *= state.shrink_leg(loop_idx, -1)
            return status

    return 0

cdef class Event_t:
    cdef np.float64_t time
    cdef np.int64_t event_idx

    def __cinit__(self, time, event_idx):
        self.time = time
        self.event_idx = event_idx

    def __richcmp__(Event_t self, Event_t other, int op):
        if op == 0:
            return 1 if self.time <  other.time else 0
        elif op == 1:
            return 1 if self.time <= other.time else 0
        elif op == 2:
            return 1 if self.time == other.time else 0
        elif op == 3:
            return 1 if self.time != other.time else 0
        elif op == 4:
            return 1 if self.time >  other.time else 0
        elif op == 5:
            return 1 if self.time >= other.time else 0

cdef class Event_heap:
    """Taken from the official Python website"""
    cdef list heap
    cdef dict entry_finder

    def __cinit__(self):
        self.heap = list()
        self.entry_finder = dict()

    cdef add_event(Event_heap self, np.int64_t event_idx, np.float64_t time=0):
        'Add a new event or update the time of an existing event.'
        if event_idx in self.entry_finder:
            self.remove_event(event_idx)
        cdef Event_t entry = Event_t(time, event_idx)
        self.entry_finder[event_idx] = entry
        heappush(self.heap, entry)

    cdef remove_event(Event_heap self, np.int64_t event_idx):
        'Mark an existing event as REMOVED.'
        cdef Event_t entry
        if event_idx in self.entry_finder:
            entry = self.entry_finder.pop(event_idx)
            entry.event_idx = -1

    cdef Event_t pop_event(Event_heap self):
        'Remove and return the closest event.'
        cdef Event_t entry
        while self.heap:
            entry = heappop(self.heap)
            if entry.event_idx != -1:
                del self.entry_finder[entry.event_idx]
                return entry
        return Event_t(0, 0.0)


cpdef simulate(p):
    '''Simulate a system of loop extruding LEFs on a 1d lattice.
    Allows to simulate two different types of LEFs, with different
    residence times and rates of backstep. This version of the simulations
    allows to simulated two-sided extrusion, i.e. type of extrusion when
    blocking one side of LEF also block the other.

    Parameters
    ----------
    p : a dictionary with parameters
        PROCESS_NAME : the title of the simulation
        L : the number of sites in the lattice
        N : the number of LEFs
        R_EXTEND : the rate of loop extension,
            can be set globally with a float,
            or individually with an array of floats
        R_SHRINK : the rate of LEF backsteps,
            can be set globally with a float,
            or individually with an array of floats
        R_OFF : the rate of detaching from the polymer,
            can be set globally with a float,
            or individually with an array of floats
        INIT_L_SITES : the initial positions of the left legs of the LEFs,
                       If -1, the position of the LEF is chosen randomly,
                       with both legs next to each other. By default is -1 for
                       all LEFs.
        INIT_R_SITES : the initial positions of the right legs of the LEFs
        ACTIVATION_TIMES : the times at which the LEFs enter the system.
            By default equals 0 for all LEFs.
            Must be 0 for the LEFs with defined INIT_L_SITES
            and INIT_R_SITES.

        ONE_SIDE_EXTEND : if True, both LEF legs extrude loops
                          independently. Otherwise, if one is blocked, the other
                          stops loop extrusion too.
        T_MAX : the duration of the simulation
        N_SNAPSHOTS : the number of time frames saved in the output. The frames
                      are evenly distributed between 0 and T_MAX.

    '''
    cdef char* PROCESS_NAME = p['PROCESS_NAME']

    cdef np.int64_t L = p['L']
    cdef np.int64_t N = np.round(p['N'])
    cdef np.float64_t T_MAX = p['T_MAX']
    cdef np.int64_t N_SNAPSHOTS = p['N_SNAPSHOTS']
    cdef np.int64_t ONE_SIDE_EXTEND = p['ONE_SIDE_EXTEND']

    cdef np.int64_t i

    cdef np.float64_t [:] RS_EXTEND = np.zeros(N, dtype=np.float64)
    cdef np.float64_t [:] RS_SHRINK = np.zeros(N, dtype=np.float64)
    cdef np.float64_t [:] RS_OFF = np.zeros(N, dtype=np.float64)

    for i in range(N):
        RS_EXTEND[i] = p['R_EXTEND'][i] if type(p['R_EXTEND']) in (list, np.ndarray) else p['R_EXTEND']
        RS_SHRINK[i] = p['R_SHRINK'][i] if type(p['R_SHRINK']) in (list, np.ndarray) else p['R_SHRINK']
        RS_OFF[i] = p['R_OFF'][i] if type(p['R_OFF']) in (list, np.ndarray) else p['R_OFF']

    cdef np.int64_t [:] INIT_L_SITES = p.get('INIT_L_SITES',
        (-1) * np.ones(N, dtype=np.int64))
    cdef np.int64_t [:] INIT_R_SITES = p.get('INIT_R_SITES',
        (-1) * np.ones(N, dtype=np.int64))

    cdef np.float64_t [:] ACTIVATION_TIMES = p.get('ACTIVATION_TIMES',
        np.zeros(N, dtype=np.float64))

    for i in range(N):
        if INIT_L_SITES[i] != -1:
            assert (INIT_R_SITES[i] != -1)
            assert ACTIVATION_TIMES[i] == 0
        else:
            assert (INIT_R_SITES[i] == -1)

    cdef State state = State(L, N, INIT_L_SITES, INIT_R_SITES)

    cdef np.int64_t [:,:] l_sites_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.int64_t [:,:] r_sites_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.float64_t [:] ts_traj = np.zeros(N_SNAPSHOTS, dtype=np.float64)

    cdef np.int64_t last_event = 0

    cdef np.float64_t time = 0
    cdef np.float64_t prev_snapshot_t = 0
    cdef np.float64_t tot_rate = 0
    cdef np.int64_t snapshot_idx = 0

    cdef Event_heap evheap = Event_heap()
    # Move LEFs onto the lattice at the corresponding activations times.
    # If the positions were predefined, initialize the fall-off time in the
    # standard way.
    for i in range(state.N):
        if (INIT_L_SITES[i] == -1) and (INIT_R_SITES[i] == -1):
            evheap.add_event(i + 2 * state.N, ACTIVATION_TIMES[i])
        else:
            evheap.add_event(i + 2 * state.N, np.random.exponential(1.0 / RS_OFF[i]))

    cdef Event_t event
    cdef np.int64_t event_idx

    while snapshot_idx < N_SNAPSHOTS:
        for i in state.recently_updated:
            evheap.remove_event(i)
            evheap.remove_event(i + state.N)
            if ONE_SIDE_EXTEND:
                if (state.l_extendable[i] + state.r_extendable[i]) > 0:
                    evheap.add_event(i, time + rand_exp(1.0 / RS_EXTEND[i]))
                if (RS_SHRINK[i] > 0) and (state.l_shrinkable[i] + state.r_shrinkable[i] > 0):
                    evheap.add_event(i + state.N, time + rand_exp(1.0 / RS_SHRINK[i]))
            else:
                if state.l_extendable[i] * state.r_extendable[i] > 0:
                    evheap.add_event(i, time + rand_exp(1.0 / RS_EXTEND[i]))
                if (RS_SHRINK[i] > 0) and (state.l_shrinkable[i] * state.r_shrinkable[i] > 0):
                    evheap.add_event(i + state.N, time + rand_exp(1.0 / RS_SHRINK[i]))

        state.reset_recently_updated()
        event = evheap.pop_event()
        time = event.time
        event_idx = event.event_idx

        if event_idx < state.N:
            last_event = extend_loop(state, event_idx, ONE_SIDE_EXTEND)
        elif event_idx < 2 * state.N:
            last_event = shrink_loop(state, event_idx - state.N, ONE_SIDE_EXTEND)
        else:
            last_event = move_loop(state, event_idx - 2 * state.N)
            evheap.add_event(event_idx,
                time + rand_exp(1.0 / RS_OFF[event_idx - 2 * state.N]))

        #if not state.check_state():
        #    print 'the state check has failed'
        #    return 0

        if last_event == 0:
            print 'an assertion failed somewhere'
            return 0

        if time > prev_snapshot_t + T_MAX / N_SNAPSHOTS:
            prev_snapshot_t = time
            l_sites_traj[snapshot_idx] = state.l_sites
            r_sites_traj[snapshot_idx] = state.r_sites
            ts_traj[snapshot_idx] = time
            snapshot_idx += 1
            if snapshot_idx % 10 == 0:
                print PROCESS_NAME, snapshot_idx, time, T_MAX
            np.random.seed()

    return np.array(l_sites_traj), np.array(r_sites_traj), np.array(ts_traj)


