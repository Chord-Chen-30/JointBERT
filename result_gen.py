import sys; sys.path.append('..')
from pandas import DataFrame
import glob
import pickle

# import time
# from itertools import product
# from scripts_gen import extra_constraint
# import os

log_path = './log-0704-freqran/'
args_path = './scripts-0704-freqran/args.pkl'

args = pickle.load(open(args_path, 'rb'))
# print(args.keys())

args_txt_path = args_path.replace("pkl", "txt")
with open(args_txt_path) as f:
    lines = f.readlines()

    for i in lines:
        print(i.strip())

columns = list(args.keys()) + ['dev_acc_mean', 'test_acc_mean', 'dev_acc_std', 'test_acc_std']

columns.pop(columns.index('log_path'))
columns.pop(columns.index('seed'))
columns.pop(columns.index('task'))
columns.pop(columns.index('model_type'))
# columns.pop(columns.index('sub_task'))

columns.pop(columns.index('model_dir'))
columns.pop(columns.index('do_train'))
columns.pop(columns.index('do_eval'))
columns.pop(columns.index('use_crf'))
columns.pop(columns.index('logging_step'))

columns.pop(columns.index('verbose'))
columns.pop(columns.index('early_stop'))



def gen_table(log_path, save_path):
    global columns
    global args

    data = {}
    for c in columns:
        data[c] = []

    all_count = 0

    path = log_path

    all_pkl_files = glob.glob(path + '*.pkl')
    # list of all txt files of methods M
    all_log_num = len(all_pkl_files)
    for j, pkl_path in enumerate(all_pkl_files):
        if j % 100 == 0:
            print('\r' + str(round(j / all_log_num * 100, 2)) + '%', pkl_path, end="")

        # print(pkl_path)
        logger = pickle.load(open(pkl_path, 'rb'))

        for c in columns:
            if ('acc' not in c) and (c!="png_save_path"):
                # if c == 'replace_strategy':
                #     print(logger.hyper_params[c])
                if c == 'learning_rate':
                    data[c].append(logger.hyper_params['lr'])
                else:
                    data[c].append(logger.hyper_params[c])


            elif c == 'dev_acc_mean':
                data[c].append(logger.dev_acc_mean)
            elif c == 'test_acc_mean':
                data[c].append(logger.test_acc_mean)
            elif c == 'dev_acc_std':
                data[c].append(logger.dev_acc_std)
            elif c == 'test_acc_std':
                data[c].append(logger.test_acc_std)
            elif c == 'png_save_path':
                # data[c].append(logger.png_save_path) # TODO remove this line later
                try:
                    data[c].append(logger.png_save_path)
                except AttributeError or KeyError:
                    data[c].append('-')

            else:
                print("column: {0} not in log".format(c))
                exit(-1)

        all_count += 1


    print('\n')

    for k in data.keys():
        print(k, len(data[k]))

    dataframe = DataFrame(data=data, columns=columns)

    pickle.dump(dataframe, open(save_path + 'table.df', 'wb'))
    print("\ndataframe generated !")

    # dataframe.to_csv(save_path + 'table.csv', float_format='%11.6f')
    # print("csv generated")

    dataframe.to_excel(save_path + 'table.xls')
    print("excel generated")

    print("Total {0} logs read.".format(all_count))




if __name__ == "__main__":

    check = input("reading from log dir: {0}, \n             script dir: {1} quit(q) \n".format(log_path, args_path))
    if check == 'q':
        exit(0)

    save_path = log_path
    dataframe_path = save_path + 'table.df'

    gen_table(log_path, save_path)

    # I use Jupyter instead for now
    # print("Generating brief result...")
    # gen_brief_table_from_excel(dataframe_path, save_path)
    # print("brief result done !")
