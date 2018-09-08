import numpy as np

#### utilities ####

def load_pickle(filename): 
    return pickle.load(open( filename,'rb'  ) )


#### boundary_list independent statistics #####
#        requires:
#            LEF_array (Mx2 numpy array of monomer positions), 
#            polymer_length (integer)

def calc_avg_coverage_by_LEFs(filelist, polymer_length, loadFunction=load_pickle):
    '''
        calculates the average coverage (fraction of polymer covered by at least one loop)
        averages over files in filelist
    '''
    avgCov = []
    for f in filelist:
        try:
            LEF_array  = np.array(loadFunction(f),dtype=int)
            loopCoverage = calc_coverage_by_LEFs(LEF_array, polymer_length)
            avgCov.append( loopCoverage)
        except:
            print('bad file', f)
            continue
    return np.mean(avgCov)

def calc_coverage_by_LEFs(LEF_array, polymer_length):
    '''
        calculates the average coverage (fraction of polymer covered by at least one loop)
    '''
    loopCoverage = np.zeros((polymer_length, ))
    for p in LEF_array: loopCoverage[p[0]:p[1]+1] += 1
    return  np.sum(loopCoverage>0) / (0.0+polymer_length)


#### boundary_list statistics #####
#        requires:
#            LEF_array (Mx2 numpy array of monomer positions), 
#            polymer_length (integer), 
#            boundary_list (Kx1 array of monomer positions)


def calc_average_LEF_stalling(filelist, polymer_length, boundary_list, loadFunction=load_pickle):
    '''
        calculates the average fraction of LEFs with two, one, or no legs stalled at boundaries.
        averages over files in filelist
    '''
    both_one_none = []
    for f in filelist:
        try:
            LEF_array = loadFunction(f)
            both_one_none.append(calc_LEF_stalling_by_leg(LEF_array, polymer_length, boundary_list))        
        except:
            print('bad file', f)
            continue
    return np.mean(both_one_none,axis=0)

def calc_LEF_stalling_by_leg(LEF_array, polymer_length, boundary_list):
    '''
        calculates the fraction of LEFs with two, one, or no legs stalled at boundaries.
        returns the fraction with both legs, one leg, or neither leg overlapping boundary_list
    '''
    if LEF_array.shape[1] !=2: raise Exception('needs to be a numLEFs x 2 array of LEFs')
    boundary_list = np.array(boundary_list,dtype=int).flatten()
    isBoundary =  np.histogram(boundary_list,  np.arange(0,polymer_length+1))[0]
    LEF_arm_status = np.sum( isBoundary[LEF_array] , 1)
    return  [  (np.sum(LEF_arm_status==2)/len(LEF_array) ),  (np.sum(LEF_arm_status==1)/len(LEF_array) ) ,(np.sum(LEF_arm_status==0)/len(LEF_array) )  ] 








