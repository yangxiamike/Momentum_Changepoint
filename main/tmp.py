import argparse

parser = argparse.ArgumentParser(description='姓名')
# parser.add_argument('--family', type=str, help='姓')
# parser.add_argument('--name', type=str, required=True, default='', help='名')
parser.add_argument('-i', '--input_dir_path', type = str, required = True, help = 'Input directory path')
parser.add_argument('-o', '--output_dir_path', type = str, required = True, help = 'Output directory path')
parser.add_argument('-c', '--use_columns', type = list, default = ['date', 'price'], 
                    nargs = '+', help = 'Use columns for price data')
parser.add_argument('-n', '--names', type = list, default = ['date', 'price'],
                    nargs = '+', help = 'Column name for price and date')
verv = parser.parse_args()
import pdb; pdb.set_trace()

#打印姓名
a=args.input_dir_path+args.output_dir_path
print(a)