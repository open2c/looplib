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
    return np.random.exponential(mean) if mean > 0 else 0

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

cdef class System:
    cdef np.int64_t L
    cdef np.int64_t N
    cdef np.float64_t time

    cdef np.float_t [:] vels
    cdef np.float_t [:] lifespans
    cdef np.float_t [:] rebinding_times
    cdef np.float_t [:] perms
    cdef np.float_t [:] centers

    # the occupancy of each site of the system. If -1, the site is unoccupied.
    # If between 0 and 2*N, corresponds to a leg of a LEF
    # If between 2*N and 4*N, corresponds to the center of a paired LEF.
    cdef np.int64_t [:] lattice
    cdef np.int64_t [:] locs

    def __cinit__(self, L, N, vels, lifespans, rebinding_times,
                  init_locs=None, perms=None):
        self.L = L
        self.N = N
        self.lattice = -1 * np.ones(L, dtype=np.int64)
        self.locs = -1 * np.ones(4*N, dtype=np.int64)
        self.vels = vels
        self.lifespans = lifespans
        self.rebinding_times = rebinding_times

        if perms is None:
            self.perms = np.ones(L+1, dtype=np.float64)
        else:
            self.perms = perms

        self.perms[0] = 0.0
        self.perms[-1] = 0.0

        cdef np.int64_t i

        # Initialize non-random loops
        for i in range(4 * self.N):
            # Check if the loop is preinitialized.
            if (init_locs[i] < 0):
                continue

            # Populate a site.
            self.locs[i] = init_locs[i]
            self.lattice[self.locs[i]] = i

    cdef np.int64_t make_step(System self, np.int64_t leg_idx, np.int64_t direction):
        """
        The variable `direction` can only take values +1 or -1.
        """

        cdef np.int64_t new_pos = self.locs[leg_idx] + direction
        return self.move_leg(leg_idx, new_pos)

    cdef np.int64_t move_leg(System self, np.int64_t leg_idx, np.int64_t new_pos):
        if (new_pos >= 0) and (self.lattice[new_pos] >=0):
            return 0

        cdef np.int64_t prev_pos

        prev_pos = self.locs[leg_idx]
        self.locs[leg_idx] = new_pos

        if prev_pos >= 0:
            if self.lattice[prev_pos] < 0:
                return 0
            self.lattice[prev_pos] = -1

        if new_pos >= 0:
            self.lattice[new_pos] = leg_idx

        return 1

    cdef np.int64_t check_system(System self):
        okay = 1
        cdef np.int64_t i
        for i in range(self.N):
            if (self.locs[i] == self.locs[i+self.N]):
                print('loop ' , i, 'has both legs at ', self.locs[i])
                okay = 0
            if (self.locs[i] >= self.L):
                print('leg ', i, 'is located outside of the system: ', self.locs[i])
                okay = 0
            if (self.locs[i+self.N] >= self.L):
                print('leg ', i+self.N, 'is located outside of the system: ', self.locs[i+self.N])
                okay = 0
            if (((self.locs[i] < 0) and (self.locs[i+self.N] >= 0 ))
                or ((self.locs[i] >= 0) and (self.locs[i+self.N] < 0 ))):
                print('the legs of the loop', i, 'are inconsistent: ', self.locs[i], self.locs[i+self.N])
                okay = 0
        return okay

cdef class Event_t:
    cdef public np.float64_t time
    cdef public np.int64_t event_idx

    def __cinit__(Event_t self, np.float_t time, np.int64_t event_idx):
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
    cdef public list heap
    cdef public dict entry_finder

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


cdef regenerate_event(System system, Event_heap evheap, np.int64_t event_idx):
    """
    Regenerate an event in an event heap. If the event is currently impossible (e.g. a step
    onto an occupied site) then the new event is not created, but the existing event is not
    modified.

    Possible events:
    0 to 2N-1 : a step to the left
    2N to 4N-1 : a step to the right
    4N to 5N-1 : passive unbinding
    5N to 6N-1 : rebinding to a randomly chosen site
    """

    cdef np.int64_t leg_idx, loop_idx
    cdef np.int64_t direction
    cdef np.float_t local_vel

    # A step to the left or to the right.
    if (event_idx < 4 * system.N) :
        if event_idx < 2 * system.N:
            leg_idx = event_idx
            direction = -1
        else:
            leg_idx = event_idx - 2 * system.N
            direction = 1

        if (system.locs[leg_idx] >= 0):
            # Local velocity = velocity * permeability
            local_vel = (system.perms[system.locs[leg_idx] + (direction+1)//2]
                * system.vels[leg_idx + (direction+1) * system.N])

            if local_vel > 0:
                if system.lattice[system.locs[leg_idx] + direction] < 0:
                    evheap.add_event(
                        event_idx,
                        system.time + rand_exp(1.0 / local_vel))

    # Passive unbinding.
    elif (event_idx >= 4 * system.N) and (event_idx < 5 * system.N):
        loop_idx = event_idx - 4 * system.N
        if (system.locs[loop_idx] >= 0) and (system.locs[loop_idx+system.N] >= 0):
            evheap.add_event(
                event_idx,
                system.time + rand_exp(system.lifespans[loop_idx]))

    # Rebinding from the solution to a random site.
    elif (event_idx >= 5 * system.N) and (event_idx < 6 * system.N):
        loop_idx = event_idx - 5 * system.N
        if (system.locs[loop_idx] < 0) and (system.locs[loop_idx+system.N] < 0):
            evheap.add_event(
                event_idx,
                system.time + rand_exp(system.rebinding_times[loop_idx]))

cdef regenerate_neighbours(System system, Event_heap evheap, np.int64_t pos):
    """
    Regenerate the motion events for the adjacent loop legs.
    Use to unblock the previous neighbors and block the new ones.
    """
    # regenerate left step for the neighbour on the right
    if ((pos<system.L-1)
        and (system.lattice[pos+1] >= 0)
        and (system.lattice[pos+1] < 2*system.N)):
        regenerate_event(system, evheap, system.lattice[pos+1])

    # regenerate right step for the neighbour on the left
    if ((pos>0) 
        and (system.lattice[pos-1] >= 0)
        and (system.lattice[pos-1] < 2*system.N)):
        regenerate_event(system, evheap, system.lattice[pos-1] + 2 * system.N)
 
cdef regenerate_all_loop_events(System system, Event_heap evheap,
                                np.int64_t loop_idx):
    """
    Regenerate all possible events for a loop. Includes the four possible motions
    and passive unbinding.
    """

    regenerate_event(system, evheap, loop_idx)
    regenerate_event(system, evheap, loop_idx + system.N)
    regenerate_event(system, evheap, loop_idx + 2 * system.N)
    regenerate_event(system, evheap, loop_idx + 3 * system.N)
    regenerate_event(system, evheap, loop_idx + 4 * system.N)
    regenerate_event(system, evheap, loop_idx + 5 * system.N)


cdef np.int64_t do_event(System system, Event_heap evheap, np.int64_t event_idx) except 0:
    """
    Apply an event from a heap on the system and then regenerate it.
    If the event is currently impossible (e.g. a step onto an occupied site),
    it is not applied, however, no warning is raised.

    Also, partially checks the system for consistency. Returns 0 is the system
    is not consistent (a very bad sign), otherwise returns 1 if the event was a step
    and 3 if the event was rebinding.

    Possible events:
    0 to 2N-1 : a step to the left
    2N to 4N-1 : a step to the right
    4N to 5N-1 : passive unbinding
    5N to 6N-1 : rebinding to a randomly chosen site
    """

    cdef np.int64_t status
    cdef np.int64_t new_pos, leg_idx, prev_pos, direction, loop_idx

    if event_idx < 4 * system.N:
        # Take a step
        if event_idx < 2 * system.N:
            leg_idx = event_idx
            direction = -1
        else:
            leg_idx = event_idx - 2 * system.N
            direction = 1

        prev_pos = system.locs[leg_idx]
        # check if the loop was attached to the chromatin
        status = 1
        if (prev_pos >= 0):
            # make a step only if there is no boundary and the new position is unoccupied
            if (system.perms[prev_pos + (direction + 1) // 2] > 0):
                if system.lattice[prev_pos+direction] < 0:
                    status *= system.make_step(leg_idx, direction)
                    # regenerate events for the previous and the new neighbors
                    regenerate_neighbours(system, evheap, prev_pos)
                    regenerate_neighbours(system, evheap, prev_pos+direction)

            # regenerate the performed event
            regenerate_event(system, evheap, event_idx)

    elif (event_idx >= 4 * system.N) and (event_idx < 5 * system.N):
        # rebind the loop
        loop_idx = event_idx - 4 * system.N

        status = 2
        # check if the loop was attached to the chromatin
        if ((system.locs[loop_idx] < 0) 
            or (system.locs[loop_idx+system.N] < 0)
            or (system.locs[loop_idx+2*system.N] < 0) 
            or (system.locs[loop_idx+3*system.N] < 0)):
            status = 0

        # save previous positions, but don't update neighbours until the loop
        # has moved
        prev_pos1 = system.locs[loop_idx]
        prev_pos2 = system.locs[loop_idx + system.N]
        prev_pos3 = system.locs[loop_idx + 2*system.N]
        prev_pos4 = system.locs[loop_idx + 3*system.N]

        status *= system.move_leg(loop_idx, -1)
        status *= system.move_leg(loop_idx+system.N, -1)
        status *= system.move_leg(loop_idx+2*system.N, -1)
        status *= system.move_leg(loop_idx+3*system.N, -1)

        # regenerate events for the loop itself and for its previous neighbours
        regenerate_all_loop_events(system, evheap, loop_idx)

        # update the neighbours after the loop has moved
        regenerate_neighbours(system, evheap, prev_pos1)
        regenerate_neighbours(system, evheap, prev_pos2)
        regenerate_neighbours(system, evheap, prev_pos3)
        regenerate_neighbours(system, evheap, prev_pos4)

    elif (event_idx >= 5 * system.N) and (event_idx < 6 * system.N):
        loop_idx = event_idx - 5 * system.N

        status = 2
        # check if the loop was not attached to the chromatin
        if ((system.locs[loop_idx] >= 0) 
            or (system.locs[loop_idx+system.N] >= 0)
            or (system.locs[loop_idx+2*system.N] >= 0)
            or (system.locs[loop_idx+3*system.N] >= 0)
            ):
            status = 0

        # find a new position for the LEM (a brute force method, can be
        # improved)
        while True:
            new_pos = rand_int(system.L-1)
            if (system.lattice[new_pos] < 0
                and system.lattice[new_pos+1] < 0
                and system.lattice[new_pos+2] < 0
                and system.lattice[new_pos+3] < 0
                and system.perms[new_pos+1] > 0
                and system.perms[new_pos+2] > 0
                and system.perms[new_pos+3] > 0):
                break

        # rebind the loop
        status *= system.move_leg(loop_idx, new_pos)
        status *= system.move_leg(loop_idx+system.N, new_pos+3)
        status *= system.move_leg(loop_idx+2*system.N, new_pos+1)
        status *= system.move_leg(loop_idx+3*system.N, new_pos+2)

        # regenerate events for the loop itself and for its new neighbours
        regenerate_all_loop_events(system, evheap, loop_idx)
        regenerate_neighbours(system, evheap, new_pos)
        regenerate_neighbours(system, evheap, new_pos + 1)
        regenerate_neighbours(system, evheap, new_pos + 2)
        regenerate_neighbours(system, evheap, new_pos + 3)
    else:
        print('event_idx assumed a forbidden value :', event_idx)

    return status


cpdef simulate(p, verbose=True):
    '''Simulate a system of loop extruding LEFs on a 1d lattice.
    Allows to simulate two different types of LEFs, with different
    residence times and rates of backstep.

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
        REBINDING_TIME : the average time that a LEF spends in solution before
            rebinding to the polymer; can be set globally with a float,
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

        T_MAX : the duration of the simulation
        N_SNAPSHOTS : the number of time frames saved in the output. The frames
                      are evenly distributed between 0 and T_MAX.

    '''
    cdef char* PROCESS_NAME = p['PROCESS_NAME']

    cdef np.int64_t L = p['L']
    cdef np.int64_t N = np.round(p['N'])
    cdef np.float64_t T_MAX = p['T_MAX']
    cdef np.int64_t N_SNAPSHOTS = p['N_SNAPSHOTS']

    cdef np.int64_t i

    cdef np.float64_t [:] VELS = np.zeros(4*N, dtype=np.float64)
    cdef np.float64_t [:] LIFESPANS = np.zeros(N, dtype=np.float64)
    cdef np.float64_t [:] REBINDING_TIMES = np.zeros(N, dtype=np.float64)

    for i in range(N):
        VELS[i] =   VELS[i+3*N] = p['R_EXTEND'][i] if type(p['R_EXTEND']) in (list, np.ndarray) else p['R_EXTEND']
        VELS[i+N] = VELS[i+2*N] = p['R_SHRINK'][i] if type(p['R_SHRINK']) in (list, np.ndarray) else p['R_SHRINK']
        LIFESPANS[i] = (1.0 / p['R_OFF'][i]) if type(p['R_OFF']) in (list, np.ndarray) else 1.0 / p['R_OFF']
        REBINDING_TIMES[i] = (
            (p['REBINDING_TIME'][i])
            if type(p.get('REBINDING_TIME',0)) in (list, np.ndarray)
            else p.get('REBINDING_TIME',0))

    cdef np.int64_t [:] INIT_LOCS = (-1) * np.ones(4*N, dtype=np.int64)
    if ('INIT_L_SITES' in p) and ('INIT_R_SITES' in p):
        for i in range(N):
            INIT_LOCS[i] = p['INIT_L_SITES'][i]
            INIT_LOCS[i+N] = p['INIT_R_SITES'][i]

    cdef np.float64_t [:] ACTIVATION_TIMES = p.get('ACTIVATION_TIMES',
        np.zeros(N, dtype=np.float64))

    for i in range(N):
        if INIT_LOCS[i] != -1:
            assert (INIT_LOCS[i+N] != -1)
            assert ACTIVATION_TIMES[i] == 0
        else:
            assert (INIT_LOCS[i+N] == -1)

    cdef np.float_t [:] PERMS = p.get('PERMS', None)
    if (not (PERMS is None)) and (PERMS.size != L+1):
        raise Exception(
            'The length of the provided array of permeabilities should be L+1')

    cdef System system = System(L, N, VELS, LIFESPANS, REBINDING_TIMES, INIT_LOCS, PERMS)

    cdef np.int64_t [:,:] l_sites_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.int64_t [:,:] r_sites_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.float64_t [:] ts_traj = np.zeros(N_SNAPSHOTS, dtype=np.float64)

    cdef np.int64_t last_event = 0

    cdef np.float64_t prev_snapshot_t = 0
    cdef np.float64_t tot_rate = 0
    cdef np.int64_t snapshot_idx = 0

    cdef Event_heap evheap = Event_heap()

    # Move LEFs onto the lattice at the corresponding activations times.
    # If the positions were predefined, initialize the fall-off time in the
    # standard way.
    for i in range(system.N):
        # if the loop location is not predefined, activate it
        # at the predetermined time
        if (INIT_LOCS[i] == -1) and (INIT_LOCS[i] == -1):
            evheap.add_event(i + 5 * system.N, ACTIVATION_TIMES[i])

        # otherwise, the loop is already placed on the lattice and we need to
        # regenerate all of its events and the motion of its neighbours
        else:
            regenerate_all_loop_events(system, evheap, i)
            regenerate_neighbours(system, evheap, INIT_LOCS[i])
            regenerate_neighbours(system, evheap, INIT_LOCS[i+system.N])

    cdef Event_t event
    cdef np.int64_t event_idx

    while snapshot_idx < N_SNAPSHOTS:
        event = evheap.pop_event()
        system.time = event.time
        event_idx = event.event_idx

        status = do_event(system, evheap, event_idx)

        if status == 0:
            print('an assertion failed somewhere')
            return 0

        if system.time > prev_snapshot_t + T_MAX / N_SNAPSHOTS:
            prev_snapshot_t = system.time
            l_sites_traj[snapshot_idx] = system.locs[:N]
            r_sites_traj[snapshot_idx] = system.locs[N:]
            ts_traj[snapshot_idx] = system.time
            snapshot_idx += 1
            if verbose and (snapshot_idx % 10 == 0):
                print(PROCESS_NAME, snapshot_idx, system.time, T_MAX)
            np.random.seed()

    return np.array(l_sites_traj), np.array(r_sites_traj), np.array(ts_traj)
