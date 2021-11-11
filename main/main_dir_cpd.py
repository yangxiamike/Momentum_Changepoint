import multiprocessing
import os
from settings.default import CPD_INPUT_FOLER_DEFAULT
from settings.default import CPD_OUTPUT_FOLDER_DEFAULT
from settings.default import START_DATE
from settings.default import END_DATE
from settings.default import CPD_DEFAULT_LBW


file_tickers = os.listdir(CPD_INPUT_FOLER_DEFAULT)
file_tickers = [file for file in file_tickers if '.csv' in file]
tickers = [os.path.basename(file).split('.')[0] for file in file_tickers]
N_WORKERS = min(len(tickers), 8)

cpd_score_file_path = f"{CPD_OUTPUT_FOLDER_DEFAULT}/cpd_score"
cpd_detect_file_path = f"{CPD_OUTPUT_FOLDER_DEFAULT}/cpd_detect"
if not os.path.exists(cpd_score_file_path):
    os.makedirs(cpd_score_file_path)
if not os.path.exists(cpd_detect_file_path):
    os.makedirs(cpd_detect_file_path)

all_processes = []
for ticker, file in zip(tickers, file_tickers):
    data_input_path = os.path.join(CPD_INPUT_FOLER_DEFAULT, file)
    data_score_path = os.path.join(cpd_score_file_path, file)
    data_detect_path = os.path.join(cpd_detect_file_path, file)
    all_processes.append(f'python main/main_single_cpd.py {data_input_path} {data_score_path} {data_detect_path} {START_DATE} {END_DATE} {CPD_DEFAULT_LBW}')

if __name__ == '__main__':
    num_proces = len(all_processes)
    for i in range(0, num_proces, 2):
        procs = all_processes[i : i+2]
        process_pool = multiprocessing.Pool(processes=2)
        process_pool.map(os.system, procs)

    # from main_single_cpd import main
    # from settings.default import CPD_DEFAULT_LBW
    # main(data_input_path, data_score_path, data_detect_path, START_DATE, END_DATE, CPD_DEFAULT_LBW)