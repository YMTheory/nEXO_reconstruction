# Fitting analysis directory

Analysis scripts for fitting results.

## Configuration

- labels.yml: this is a fixed ymal configuration file where I specify the label name for variable name. (Do not need to change)!

- config.csv: this is the file I specify which fitting configurations I wanna use to do the analysis. For each set, a new row needs to be specify following the entry names listed as the column header.

## Tools

- loadFits.py: matching fitting result files for the certain configuration (load from config.csv) and reading, merging variables I want to analyse.

- draw.py: plotting 1D, 2D... distributions of different fitting variables; plotting fitting waveforms for single events.

## Main entry:

- analysis.py: where I do the whole analysis using the configuration files and tool files. I need to specify a list of variables I want to analyse, currently it will print the charge reconstruction and plot all 1D and 2D plots for thoes variables.

An example running command could be `python analysis.py --variables fitx fity relQ`


- waveform.py: plot fitting waveforms for single event and calculate the collected charges on each strip of this event. Currently the single fitting event is randomly selected from the fitting results csv file.

An exmple running command could be `python waveform.py`
