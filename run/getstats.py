import sys
import argparse

from raster.lyft.data.preprocess import DataAnalyser

parser = argparse.ArgumentParser(description='Manage running job')
parser.add_argument('--split', help='split of the data to analyse')
parser.add_argument('--data', default='~/lyft', help='dataset folder')
parser.add_argument('--config', default='./config.yaml', help='path to config file')
parser.add_argument('--out', default='./config.yaml', help='output directory path')
parser.add_argument('--step', default=10, type=int, help='iteration step size')

args = parser.parse_args()

if __name__ == '__main__':
    # initializing various parts
    da = DataAnalyser(data_root=args.data, config_path=args.config, split=args.split, output_folder=args.out)
    da.process(step=args.step)
