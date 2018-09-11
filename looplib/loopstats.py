import numpy as np
from inspect import getfullargspec
import pandas as pd

#### utilities ####

def load_pickle(filename): 
    return pickle.load(open( filename,'rb'  ) )

stat_output_dict = {} # create a dictionary of statistic outputs, to pass to the report

#### boundary_list independent statistics #####
#        requires:
#            LEF_array (Mx2 numpy array of monomer positions), 
#            polymer_length (integer)

def calc_coverage_by_LEFs(LEF_array, polymer_length):
    ''' calculates the average coverage (fraction of polymer covered by at least one loop)    '''
    loopCoverage = np.zeros((polymer_length, ))
    for p in LEF_array: loopCoverage[p[0]:p[1]+1] += 1
    return  np.sum(loopCoverage>0) / (0.0+polymer_length)
stat_output_dict[calc_coverage_by_LEFs] = 'coverage'

def calc_loop_size(LEF_array,polymer_length):
    return np.mean( LEF_array[:,1]-LEF_array[:,0] )
stat_output_dict[calc_loop_size] = 'loop_size'

#### boundary_list statistics #####
#        requires:
#            LEF_array (Mx2 numpy array of monomer positions), 
#            polymer_length (integer), 
#            boundary_list (Kx1 array of monomer positions)


def calc_LEF_stalling_by_leg(LEF_array, polymer_length, boundary_list):
    ''' calculates the fraction of LEFs with two, one, or no legs stalled at boundaries.
        returns the fraction with both legs, one leg, or neither leg overlapping boundary_list    '''
    if LEF_array.shape[1] !=2: raise Exception('needs to be a numLEFs x 2 array of LEFs')
    boundary_list = np.array(boundary_list,dtype=int).flatten()
    isBoundary =  np.histogram(boundary_list,  np.arange(0,polymer_length+1))[0]
    LEF_arm_status = np.sum( isBoundary[LEF_array] , 1)
    return  [(np.sum(LEF_arm_status==2)/len(LEF_array) ),  (np.sum(LEF_arm_status==1)/len(LEF_array) ) ,(np.sum(LEF_arm_status==0)/len(LEF_array)  )]
stat_output_dict[ calc_LEF_stalling_by_leg] = ['both_stalled','one_stalled','none_stalled']

def calc_boundary_occupancy(LEF_array, polymer_length, boundary_list):
    ''' calculates the fraction of LEFs with two, one, or no legs stalled at boundaries.
        returns the fraction with both legs, one leg, or neither leg overlapping boundary_list  '''
    if LEF_array.shape[1] !=2: raise Exception('needs to be a numLEFs x 2 array of LEFs')
    boundary_list = np.array(boundary_list,dtype=int).flatten()
    bb = np.arange(0,polymer_length+1, 1); bb_mids = .5*(bb[:-1]+bb[1:])
    extruderHistogram, b = np.histogram( LEF_array ,bb)
    boundary_occupancy = np.mean( extruderHistogram[boundary_list ] )
    non_boundary_list = np.setdiff1d(np.arange(1,polymer_length-1), boundary_list)
    non_occupancy = np.mean( extruderHistogram[non_boundary_list ]  )
    return [boundary_occupancy, non_occupancy]
stat_output_dict[ calc_boundary_occupancy] = ['boundary_occupancy','non_occupancy']



##### averaging and summarizing ####

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l] # since list(itertools.chain.from_iterable(newlist)) doesn't quite cut it...

def _calc_stats(filelist, stat_function_list=[calc_coverage_by_LEFs], load_function=load_pickle, **kwargs):
    ''' takes  a list of filenames returns loop statistics over these files. can be  parallelized in the future...'''
    stat_list = []
    for f in filelist:
        try:
            LEF_array  = np.array(load_function(f),dtype=int)
            filestats = []
            for stat_function in stat_function_list:
                numInputs = len( getfullargspec(stat_function)[0])
                if numInputs  == 2:
                    filestats.append( stat_function(LEF_array, kwargs['polymer_length']))
                elif numInputs == 3:
                    filestats.append( stat_function(LEF_array, kwargs['polymer_length'], kwargs['boundary_list']))
            stat_list.append(flatten(filestats))
        except:
           print('bad file', f);   continue
    return stat_list


def _create_loopstats_report(filelist_tuples, stat_function_list=[calc_coverage_by_LEFs], 
                             load_function=load_pickle,roundDecimals=2, **kwargs):
    ''' averages the stat functions over each group of filelists; kwargs needed depend on functions called
      usage: _create_loopstats_report([(filelist1,'smclife1'), (filelist2,'smclife2') ], 
                                        stat_function_list=[calc_loop_size,calc_coverage_by_LEFs, calc_LEF_stalling_by_leg, calc_boundary_occupancy], 
                                        load_function= load_pickle, roundDecimals=2, 
                                         **{'polymer_length': polymer_length , 'boundary_list':boundaries_all}  )     '''
    loopstats_report = []
    loopstats_indicies =[]
    for filelist, filename in filelist_tuples:
        loopstats_indicies.append(filename)
        loopstats_report.append(  np.mean( _calc_stats(filelist, stat_function_list, load_function, **kwargs),axis = 0) )
    loopstats_report = np.array(loopstats_report) 
    if roundDecimals != None: loopstats_report = np.round(loopstats_report,roundDecimals)

    column_names = []
    for stat_function in stat_function_list:
        column_names.append( stat_output_dict[stat_function])
    column_names = flatten(column_names)

    loopstats_report = pd.DataFrame( loopstats_report, columns = column_names)#, index=loopstats_indicies)
    loopstats_report['row_names'] = loopstats_indicies
    loopstats_report.set_index('row_names',inplace=True, drop=True)
    return loopstats_report





