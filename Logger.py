from datetime import datetime, timezone, timedelta
import random
import pickle
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt
import os

class MyLogger(object):

    def __init__(self, save_path, args, cmd):

        try:
            mkdir(os.path.dirname(save_path))
        except FileExistsError:
            pass
        
        self.args = args
        self.cmd = cmd
        self.save_path = save_path
        self.hyper_params = {}
        self.logs = []
        self.result = {'best_dev_acc': [],
                       'best_test_acc': [],
                       }

        self.dev_acc_mean = 0
        self.test_acc_mean = 0
        self.dev_acc_std = 0
        self.test_acc_std = 0

        self.seeds = args.seed.split(":")

        self.train_acc = {s: [] for s in self.seeds}
        self.dev_acc = {s: [] for s in self.seeds}
        self.test_acc = {s: [] for s in self.seeds}

        self.loss_origin = {s: [] for s in self.seeds}
        self.loss_attr = {s: [] for s in self.seeds}

        time = datetime.now(timezone.utc) + timedelta(hours=8)  # Convert UTC to Beijing
        self.time = time.strftime('%Y-%m-%d_%H.%M.%S') + str(round(random.random(), 3))[1:]

        self.save_path += (self.time + '.pkl')

        self.hyper_param_info('lr', args.learning_rate)
        self.hyper_param_info('train_batch_size', args.train_batch_size)
        self.hyper_param_info('eval_batch_size', args.eval_batch_size)
        self.hyper_param_info('sub_task', args.sub_task)

        self.hyper_param_info('shot', args.shot)
        self.hyper_param_info('replace', args.replace)
        self.hyper_param_info('replace_repeat', args.replace_repeat)
        self.hyper_param_info('weight', args.weight)
        self.hyper_param_info('margin', args.margin)



    def hyper_param_info(self, hyper_param, value):
        self.hyper_params[hyper_param] = value

    def log_info(self, line):
        self.logs.append(line)

    def result_info(self, result_item, result_value):
        self.result[result_item].append(result_value)
        # assert len(set(self.result['RE_test_acc'])) <= 1, print("RE acc changed?!", self.result['RE_test_acc'])

    def acc_info(self, fold, seed, acc):
        self.__getattribute__(fold)[str(seed)].append(acc)

    def save_plot_acc(self, fig_path):
        time = self.time.replace('.', '-')
        self.png_save_path = fig_path+time+'-acc-[SEED].png'
        for seed in self.seeds:

            plt.plot(self.train_acc[seed], label='Train Acc')
            plt.plot(self.dev_acc[seed], label='Dev Acc')
            plt.plot(self.test_acc[seed], label='Test Acc')
            plt.legend()
            plt.title('Acc')

            png_save_path = self.png_save_path.replace("[SEED]", seed)
            plt.savefig(png_save_path, format='png')
            plt.close()
        print("acc figure saved!")

    def append_loss_origin(self, seed, loss):
        self.loss_origin[str(seed)].append(loss)

    def append_loss_attr(self, seed, loss):
        self.loss_attr[str(seed)].append(loss)

    def save_plot_loss(self, fig_path):
        time = self.time.replace('.', '-')
        # self.png_save_path = fig_path + time + '-loss-[SEED].png'

        for seed in self.seeds:

            plt.plot(self.loss_origin[str(seed)], label='origin_loss-{0}-{1}-{2}-{3}'.format(str(seed), self.args.replace, self.args.weight, self.args.margin))
            plt.legend()
            plt.title(self.cmd)
            png_save_path = fig_path + time + '-loss-origin-[SEED].png'
            png_save_path = png_save_path.replace("[SEED]", seed)
            plt.savefig(png_save_path, format='png')
            plt.close()

            plt.plot(self.loss_attr[str(seed)], label='attr_loss-{0}-{1}-{2}-{3}'.format(str(seed), self.args.replace, self.args.weight, self.args.margin))
            plt.legend()
            plt.title(self.cmd)
            png_save_path = fig_path + time + '-loss-attr-[SEED].png'
            png_save_path = png_save_path.replace("[SEED]", seed)
            plt.savefig(png_save_path, format='png')
            plt.close()

        print("loss figure saved!")

    def show_loss(self):
        plt.plot(self.loss_origin, label='CE_loss')
        plt.show()

    def cal_std_mean(self):
        # Calculate mean across random seeds
        self.dev_acc_mean = float(mean(self.result['best_dev_acc']))
        self.test_acc_mean = float(mean(self.result['best_test_acc']))

        if len(self.result['best_dev_acc']) > 1:
            self.dev_acc_std = float(std(self.result['best_dev_acc'], ddof=1))  # ddof=1 gives sample standard variation
            self.test_acc_std = float(std(self.result['best_test_acc'], ddof=1))
            # print(self.dev_acc_std, self.test_acc_std)
        else:
            self.dev_acc_std = 0.0
            self.test_acc_std = 0.0

    def save(self):
        pickle.dump(self, open(self.save_path, 'wb'))
        print("Log saved!")

    def __str__(self):
        print(self.cmd)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_cmd(argv):
    cmd = ""

    print("\nExecuting:")
    # prefix = "CUDA_VISIBLE_DEVICES=X python"
    # cmd += prefix
    #
    # print(prefix, end=" ")
    cmd += "python "
    print("python", end=" ")
    for i in range(len(argv)):
        cmd += argv[i]
        cmd += " "
        print(argv[i], end=" ")
    cmd += '\n'
    print('\n')

    return cmd
