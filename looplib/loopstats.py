import numpy as np
from inspect import getfullargspec
import pandas as pd

import bioframe
from bioframe.tools import tsv, bedtools
def bedtools_intersect_counts(left, right, **kwargs):
    with tsv(left) as a, tsv(right) as b:
        out = bedtools.intersect(a=a.name, b=b.name,c=True)
        out.columns = list(left.columns) + ['counts']
    return out

#### utilities ####

def load_pickle(filename): 
    return pickle.load(open( filename,'rb'  ) )


def load_joblib_data(filename):
    return joblib.load(filename.replace('SMC','block'))['data']


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
    boundary_list = np.array(boundary_list,dtype=int).flatten()
    isBoundary =  np.histogram(boundary_list,  np.arange(0,polymer_length+1))[0]
    LEF_arm_status = np.sum( isBoundary[LEF_array] , 1)
    return  [(np.sum(LEF_arm_status==2)/len(LEF_array) ),  (np.sum(LEF_arm_status==1)/len(LEF_array) ) ,(np.sum(LEF_arm_status==0)/len(LEF_array)  )]
stat_output_dict[ calc_LEF_stalling_by_leg] = ['both_stalled','one_stalled','none_stalled']

def calc_boundary_occupancy(LEF_array, polymer_length, boundary_list):
    ''' calculates the fraction of LEFs with two, one, or no legs stalled at boundaries.
        returns the fraction with both legs, one leg, or neither leg overlapping boundary_list  '''
    boundary_list = np.array(boundary_list,dtype=int).flatten()
    bb = np.arange(0,polymer_length+1, 1); bb_mids = .5*(bb[:-1]+bb[1:])
    extruderHistogram, b = np.histogram( LEF_array ,bb)
    boundary_occupancy = np.mean( extruderHistogram[boundary_list ] )
    non_boundary_list = np.setdiff1d(np.arange(1,polymer_length-1), boundary_list)
    non_occupancy = np.mean( extruderHistogram[non_boundary_list ]  )
    return [boundary_occupancy, non_occupancy]
stat_output_dict[ calc_boundary_occupancy] = ['boundary_occupancy','non_occupancy']


def calc_boundary_crossing_percent(LEF_array, polymer_length, boundary_list):
    ''' calculates the fraction of LEFs with two, one, or no legs stalled at boundaries.
        returns the fraction with both legs, one leg, or neither leg overlapping boundary_list  '''
    boundary_list = np.array(boundary_list,dtype=int).flatten()
    allLocs_df = pd.DataFrame(boundary_list, columns=['start'])
    allLocs_df['end'] = boundary_list
    allLocs_df['chr'] = 13
    allLocs_df = allLocs_df[['chr','start','end']]
    
    LEF_array_df = pd.DataFrame(LEF_array,columns=['start','end'])
    LEF_array_df['chr'] = 13
    LEF_array_df= LEF_array_df[['chr','start','end']]
    LEF_array_df.sort_values(['start'], inplace=True)                    
    LEF_array_shrink_df = LEF_array_df.copy()
    LEF_array_shrink_df['start'] = LEF_array_shrink_df['start'].values+1
    LEF_array_shrink_df['end']   = np.maximum( LEF_array_shrink_df['end'].values-1, LEF_array_shrink_df['start'])
                    
    percent_crossing = np.sum((bedtools_intersect_counts( LEF_array_shrink_df, allLocs_df)['counts'] > 0)  
                            / len(LEF_array) )
    return [percent_crossing]
stat_output_dict[ calc_boundary_crossing_percent] = ['percent_crossing']

def calc_numLoops_given_boundaryPair_contact(LEF_array, polymer_length, boundary_pair_array, data):
    LEF_array_df = pd.DataFrame(LEF_array,columns=['start','end'])
    LEF_array_df['chr'] = 13
    LEF_array_df= LEF_array_df[['chr','start','end']]
    
    sep2_dists =   np.sum( (data[boundary_pair_array[:,0],:] 
                            - data[boundary_pair_array[:,1],:])**2.,axis=1 )**.5 
    sep2_contacts = boundary_pair_array[ sep2_dists < 3 ,:]

    sep2_contacts_df = pd.DataFrame(sep2_contacts,columns=['start','end'])
    sep2_contacts_df['chr'] = 13
    sep2_contacts_df= sep2_contacts_df[['chr','start','end']]
    sep2_contacts_df.sort_values(['start'], inplace=True)
    
    cbins = np.arange(0,len(LEF_array),1)
    a,b = np.histogram( bedtools_intersect_counts(sep2_contacts_df, LEF_array_df)['counts'].values  , cbins)
    a = a/np.sum(a)
    no_loops, single_loop, multiple_loops = ( a[0], a[1], np.sum(a[2:]) )    
    return [no_loops, single_loop, multiple_loops]
stat_output_dict[calc_numLoops_given_boundaryPair_contact]= ['no_loops','one_loop','two_plus_loops']


##### averaging and summarizing ####

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l] # since list(itertools.chain.from_iterable(newlist)) doesn't quite cut it...

def _calc_stats(filelist, stat_function_list=[calc_coverage_by_LEFs], load_function=load_pickle, load_function_data=None, **kwargs):
    ''' takes  a list of filenames returns loop statistics over these files. can be  parallelized in the future...'''
    stat_list = []
    for f in filelist:
        try:
            LEF_array  = np.array(load_function(f),dtype=int)
            if LEF_array.shape[1] !=2: raise Exception('needs to be a numLEFs x 2 array of LEFs')
            if load_function_data != None:
                data = np.array(load_function_data(f))
            filestats = []
            for stat_function in stat_function_list:
                numInputs = len( getfullargspec(stat_function)[0])
                if numInputs  == 2:
                    filestats.append( stat_function(LEF_array, kwargs['polymer_length']))
                elif numInputs == 3:
                    filestats.append( stat_function(LEF_array, kwargs['polymer_length'], kwargs['boundary_list']))
                elif numInputs == 4:
                    filestats.append( stat_function(LEF_array, kwargs['polymer_length'], kwargs['boundary_pair_array'], data ))
            stat_list.append(flatten(filestats))
        except:
           print('bad file', f);   continue
    return stat_list


def _create_loopstats_report(filelist_tuples, stat_function_list=[calc_coverage_by_LEFs], 
                             load_function=load_pickle, load_function_data=None,roundDecimals=2, **kwargs):
    ''' averages the stat functions over each group of filelists; kwargs needed depend on functions called
      usage: _create_loopstats_report([(filelist1,'smclife1'), (filelist2,'smclife2') ], 
                                        stat_function_list=[calc_loop_size,calc_coverage_by_LEFs, calc_LEF_stalling_by_leg, calc_boundary_occupancy], 
                                        load_function= load_pickle, roundDecimals=2, 
                                         **{'polymer_length': polymer_length , 'boundary_list':boundaries_all}  )     '''
    loopstats_report = []
    loopstats_indicies =[]
    for filelist, filename in filelist_tuples:
        loopstats_indicies.append(filename)
        loopstats_report.append(  np.mean( _calc_stats(filelist, stat_function_list, load_function, load_function_data, **kwargs),axis = 0) )
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





