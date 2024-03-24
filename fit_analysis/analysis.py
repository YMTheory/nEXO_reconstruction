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
parser.add_argument('--variables',  nargs='+',      help='A list of varibles.')
args = parser.parse_args()

variable_list = args.variables


############################################
# Load labels for variables

labels = []
with open('labels.yml', 'r') as label_file:
    filelist = yaml.safe_load(label_file)
    for varname in variable_list:
        try:
            labels.append( filelist[varname] )
        except KeyError:
            variable_list.remove(varname)
            print(f'Error: there is no variable named {varname} => removed from the variable list !!')
        except Exception as err:
            print(f"Other error {err}, {type(err)} !!!")

total_variables = len(variable_list)
fitVals_list = [ [] for _ in range(total_variables)]

############################################

# Load fitting configurations we want to compare
df_config = pd.read_csv('config.csv', ) #dtype=str)
df_config.drop(columns='fitid', inplace=True)
total_configs = df_config.shape[0]

############################################

# Load all fitting parameters required to do analysis

loaders, legends = [], []
for i in range(total_configs):

    tmp_ld = loader(*df_config.iloc[i].to_numpy())
    loaders.append( tmp_ld )
    legends.append( tmp_ld._get_label() )

    for j, varname in enumerate(variable_list):
        if varname == "relQ":
            tmp_fitQ    = tmp_ld.load_one_fitVals('fitQ')
            tmp_trueQ   = tmp_ld.load_one_fitVals('trueQ')
            tmp_var     = (tmp_fitQ - tmp_trueQ) / tmp_trueQ * 100

        else:
            tmp_var = tmp_ld.load_one_fitVals(varname)
        
        fitVals_list[j].append(tmp_var)


############################################



########## Printing and Drawing ##########
    
relQ = []
for ld in loaders:

    fitQ = ld.load_one_fitVals("fitQ")
    trueQ = ld.load_one_fitVals('trueQ')
    tmp_relQ = (fitQ - trueQ) / trueQ * 100
    relQ.append(tmp_relQ)

print('\n')
print('========== Charge reconstruction results ==========')
print(tabulate( [ [cfg, np.mean(v), np.std(v)] for cfg, v in zip(legends, relQ) ] , headers=["Configuration", "Bias [%]", "Resolution [%]"]))
print('===================================================')


d = drawer()
with plt.style.context("styles/my.mplstyle"):
    for val, lb in zip(fitVals_list, labels):
        fig, _ = d.draw_fitVals_1Dim(val, legends, lb)

    for j1 in range(total_variables):
        for j2 in range(j1+1, total_variables):
            fig, _ = d.draw_fitVals_2Dim(fitVals_list[j1], fitVals_list[j2], legends, labels[j1], labels[j2])
    




