import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tabulate import tabulate

from loadFits import loader
from draw import drawer

import argparse

############################################
# This is main entry of the whole analysis chain, where we need to specify which variables we are



parser = argparse.ArgumentParser(description="Arguments to configure fitting variables to analyse.")
parser.add_argument('--variables',  nargs='+',                              help='A list of varibles.')
parser.add_argument('--charge',     dest='charge',  action='store_true',    help='Plot all charge distributions.')
parser.add_argument('--no-charge',  dest='charge',  action='store_false',   help='Not plot all charge distributions.')
args = parser.parse_args()

variable_list = args.variables
if variable_list == None:
    variable_list = []

############################################
# Load labels for variables

var_labels = []
with open('labels.yml', 'r') as label_file:
    filelist = yaml.safe_load(label_file)
    for varname in variable_list:
        try:
            var_labels.append( filelist[varname] )
        except KeyError:
            variable_list.remove(varname)
            print(f'Error: there is no variable named {varname} => removed from the variable list !!')
        except Exception as err:
            print(f"Other error {err}, {type(err)} !!!")

if len(variable_list) != len(var_labels):
    print(f'Error: Can not match labels ({len(var_labels)}) for each variables ({len(variable_list)}) !!')
    sys.exit(-1)

total_variables = len(variable_list)

print(f'\n+++++++++++ Total {total_variables} variables are required for analysis ++++++++++')

############################################

# Load fitting configurations we want to compare
df_config = pd.read_csv('config.csv', ) #dtype=str)
df_config.drop(columns='fitid', inplace=True)
total_configs = df_config.shape[0]
print(f'\n+++++++++++ Total {total_configs} fitting configurations are required for analysis ++++++++++\n')

############################################

# Load all fitting parameters required to do analysis

## Initialize loaders according to config yaml filer
config_loaders, config_legends = [], []
for icfg in range(total_configs):
    tmp_ld = loader(*df_config.iloc[icfg].to_numpy())
    tmp_ld.successful_fitNumber()
    config_loaders.append( tmp_ld )
    config_legends.append( tmp_ld._get_label() )

## Load fitting parameters from each loader
fitVals_list = [[] for _ in range(total_variables)]
for ivar in range(total_variables):
    for icfg in range(total_configs):
        tmp_ld = config_loaders[icfg]
        if varname == 'relQ':
            tmp_fitQ    = tmp_ld.load_one_fitVals('fitQ')
            tmp_trueQ   = tmp_ld.load_one_fitVals('trueQ')
            tmp_var     = ( tmp_fitQ - tmp_trueQ ) / tmp_trueQ * 100
        elif varname == 'itgrelQ':
            tmp_fitQ    = tmp_ld.load_one_fitVals('itgQ')
            tmp_trueQ   = tmp_ld.load_one_fitVals('trueQ')
            tmp_var     = ( tmp_fitQ - tmp_trueQ ) / tmp_trueQ * 100
        else:
            tmp_var = tmp_ld.load_one_fitVals(varname)

        fitVals_list[ivar].append( tmp_var )



############################################



########## Printing and Drawing ##########
    
if args.charge:
    fitQs, itgQs, trueQs, relfitQs, relitgQs = [], [], [], [], [],
    for ld in config_loaders:
        f, t, i = ld.load_one_fitVals("fitQ"), ld.load_one_fitVals("trueQ"), ld.load_one_fitVals("itgQ")
        rf = ( f - t ) / t * 100
        ri = ( i - t ) / t * 100
        fitQs.append(f)
        trueQs.append(t)
        itgQs.append(i)
        relfitQs.append(rf)
        relitgQs.append(ri)


 
    print('\n')
    print('===================================================== Charge reconstruction results ==============================================')
    print(tabulate( [ [cfg, np.mean(v), np.std(v), np.mean(u), np.std(u)] for cfg, v, u in zip(config_legends, relfitQs, relitgQs) ] , headers=["Configuration", "Bias fitting [%]", "Resolution fitting [%]", "Bias integral [%]", "Resolution integral [%]", ]))
    print('==================================================================================================================================')
 
 
d = drawer()
with plt.style.context("styles/my.mplstyle"):
    for val, lb in zip(fitVals_list, var_labels):
        fig, _ = d.draw_fitVals_1Dim(val, config_legends, lb)
 
    for j1 in range(total_variables):
        for j2 in range(j1+1, total_variables):
            fig, _ = d.draw_fitVals_2Dim(fitVals_dict.values()[j1], fitVals_dict.values()[j2], config_legends, var_labels[j1], var_labels[j2])
 
 
    if args.charge:
        d.draw_raw_charge_distribution(fitQs, itgQs, trueQs, relfitQs, relitgQs, config_legends)
    




