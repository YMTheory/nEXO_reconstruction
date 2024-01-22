# Data generator outside the jupyter notebook
# author: miaoyu@slac.stanford.edu

import argparse

from toyMC_generator import toyMC_generator

def generate_toyMC_dataset(filename, entries, x0, y0, z0, q0, xx, yx, xy, yy):
    gen = toyMC_generator()
    gen.filename = filename
    gen.total_entries = entries
    
    gen.generate_bunches(xx, yx, xy, yy, q0, x0, y0, z0)
    gen.save_waveforms()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Toy MC dataset generator.')
    parser.add_argument('--x0', type=float, help='x position of the point charge.')
    parser.add_argument('--y0', type=float, help='y position of the point charge.')
    parser.add_argument('--z0', type=float, help='z position of the point charge.')
    parser.add_argument('--q0', type=float, help='1e5 charge of the point charge.')
    
    parser.add_argument('--xxstrip', type=float, nargs='*', help='X coordinates for all x-axis aligned strips.')
    parser.add_argument('--yxstrip', type=float, nargs='*', help='Y coordinates for all x-axis aligned strips.')
    parser.add_argument('--xystrip', type=float, nargs='*', help='X coordinates for all y-axis aligned strips.')
    parser.add_argument('--yystrip', type=float, nargs='*', help='Y coordinates for all y-axis aligned strips.')

    parser.add_argument('--filename', type=str, help='Output filename.')
    parser.add_argument('--entries', type=int, default=100, help='Count of simulated events.')

    args = parser.parse_args()
    args.filename = '/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/CurrentAndChargeSignalCalculator/toyMC_data/' + args.filename

    x_xstrip, y_xstrip, x_ystrip, y_ystrip = [], [], [], []
    for pos in args.xxstrip:
        x_xstrip.append(pos)
    for pos in args.yxstrip:
        y_xstrip.append(pos)
    for pos in args.xystrip:
        x_ystrip.append(pos)
    for pos in args.yystrip:
        y_ystrip.append(pos)

    generate_toyMC_dataset(args.filename, \
                           args.entries, \
                           args.x0, \
                           args.y0, \
                           args.z0, \
                           args.q0, \
                           x_xstrip, \
                           y_xstrip, \
                           x_ystrip, \
                           y_ystrip,\
                        )