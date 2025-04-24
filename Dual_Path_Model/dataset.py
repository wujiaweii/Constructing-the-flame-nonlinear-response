import os
import argparse
import random
import shutil
from shutil import copyfile
import numpy as np
import pandas as pd

class sampling:
    def __init__(self,num,switch_print,max_length):
        self.num=num
        self.switch_print=switch_print
        self.max_length=max_length

    def variable_interval(self,sparsetocompact):
        sampling_num = np.arange(1, self.num + 1, 1)
        sum_sampling_num = np.sum(sampling_num)
        delta = self.max_length / sum_sampling_num

        if sparsetocompact == True:
            a = self.max_length
        else:
            a = 0
        dn = 0
        index_list = []
        count = []
        for i in range(self.num):
            dn = dn + delta
            if sparsetocompact == True:
                a = a-dn
            else:
                a = a+dn

            index_list.append(int(a))
            count.append(i)
        index_list = np.array(index_list)
        if sparsetocompact == True:
            index_list = index_list[::-1]
        if self.switch_print == True:
            print('The information of sample index------------------------------------')
            print('sampling:variable')
            if sparsetocompact==True:
                print('sparsetocompact:True')
            else:
                print('sparsetocompact:False')
            count = np.array(count)
            print('index_list:', index_list.shape)
            print('---------------------------------------------')
        return index_list

    def equal_interval(self,):
        index_list=np.linspace(0,self.max_length-1,self.num,dtype=int)
        if self.switch_print == True:
            print('The information of sample index------------------------------------')
            print('sampling:equal')
            print('index_list:', index_list.shape, sep='')
            print('---------------------------------------------')
        return index_list

class num_sim_data:

    def __init__(self, sampling_type, sampling_num, sparsetocompact, switch_print,impact_window,train_path,test_path,test_mononum):
        self.sampling_type = sampling_type
        self.sampling_num = sampling_num
        self.sparsetocompact = sparsetocompact
        self.switch_print = switch_print
        self.impact_window=impact_window
        self.train_path=train_path
        self.test_path=test_path
        self.test_mononum=test_mononum

    def triandata(self, swit_printindex):
        x_total=None
        y_total=None
        x_index=None
        if self.sampling_type != None:
            Sampling = sampling(num=self.sampling_num, max_length=self.impact_window, switch_print=self.switch_print)
            if self.sampling_type == 'variable':
                x_index = Sampling.variable_interval(sparsetocompact=self.sparsetocompact)
            else:
                x_index = Sampling.equal_interval()
            if swit_printindex==True:
                print('sample index：',x_index)
        filenames=os.listdir(self.train_path)
        for i,filename in enumerate(filenames):
            path =  self.train_path+filename
            print(path)
            data_df = pd.read_csv(path, header=0,dtype=np.float32)
            x_y = data_df.to_numpy()
            x = x_y[:, 0]
            y = x_y[:, 1]
            x = np.lib.stride_tricks.sliding_window_view(x[1:], self.impact_window, axis=0)
            y = y[self.impact_window:]
            if self.sampling_type != None:
                x = x[:, x_index]
            if i == 0:
                x_total = x
                y_total = y
                print('x:', x.shape, x.dtype, type(x), '\n', 'y:', y.shape, sep='')
                continue
            print('x:', x.shape, x.dtype, type(x), '\n', 'y:', y.shape)
            x_total = np.concatenate((x_total, x), axis=0)
            y_total = np.concatenate((y_total, y), axis=0)

        x_total = x_total.astype(np.float32)
        y_total = y_total.astype(np.float32)
        x_total = x_total.reshape((x_total.shape[0], x_total.shape[1], 1))
        y_total = y_total.reshape((y_total.shape[0], 1))
        print('x_total:', x_total.shape, type(x_total), '\n', 'y_total:', y_total.shape, type(y_total), '\n', sep='')
        return x_total, y_total

    def testdata(self, ampfreqs):
        x_multifreqs=[]
        y_multifreqs=[]
        for ampfreq in ampfreqs:
            data_path = self.test_path + str(ampfreq[0]) + '_' + str(ampfreq[1]) + 'Hz.csv'
            data_df = pd.read_csv(data_path, header=0,dtype=np.float32)
            x_y = data_df.to_numpy()
            x = x_y[:, 0]
            y = x_y[:, 1]
            x = np.lib.stride_tricks.sliding_window_view(x[1:], self.impact_window, axis=0)
            y = y[self.impact_window:]
            if self.sampling_type != None:
                Sampling = sampling(num=self.sampling_num, max_length=self.impact_window, switch_print=False)
                if self.sampling_type == 'variable':
                    x_index = Sampling.variable_interval(sparsetocompact=self.sparsetocompact)
                    x = x[:, x_index]
                else:
                    x_index = Sampling.equal_interval()
                    x = x[:, x_index]
            sampling_index=np.linspace(0,len(x)-1,self.test_mononum,dtype=int)
            x_multifreqs.append(x[sampling_index,:])
            y_multifreqs.append(y[sampling_index])

        x = np.concatenate(x_multifreqs,axis=0).astype(np.float32)
        y = np.concatenate(y_multifreqs,axis=0).astype(np.float32)
        x = x.reshape((x.shape[0], x.shape[1], 1))
        y = y.reshape((y.shape[0], 1))
        return x, y

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def re_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def Dataset(config):
    Num_sim_data_equal=num_sim_data(sampling_type='equal',sampling_num=config.sampling_number,sparsetocompact=config.sparsetocompact,switch_print=config.print_index,
                              impact_window=config.impact_window,train_path=config.origin_train_data,
                              test_path=config.origin_test_data,test_mononum=config.singlefreq_test_number)
    x_train_val_equal,y_train_val_equal=Num_sim_data_equal.triandata(swit_printindex=True)
    x_test_equal,y_test_equal=Num_sim_data_equal.testdata(
        ampfreqs=[

            [0.25, 100], [0.25, 200], [0.25, 300], [0.25, 400], [0.25, 500], [0.25, 600], [0.25,700],  [0.25, 800],
            [0.25, 900],
            [0.35, 100], [0.35, 200], [0.35, 300], [0.35, 400], [0.35, 500], [0.35, 600], [0.35, 700], [0.35, 800],
            [0.35, 900],
            [0.45, 100], [0.45, 200], [0.45, 300], [0.45, 400], [0.45, 500], [0.45, 600], [0.45, 700], [0.45, 800],
            [0.45, 900],
            [0.55, 100], [0.55, 200], [0.55, 300], [0.55, 400], [0.55, 500], [0.55, 600], [0.55, 700], [0.55, 800],
            [0.55, 900],
            [0.65, 100], [0.65, 200], [0.65, 300], [0.65, 400], [0.65, 500], [0.65, 600], [0.65, 700], [0.65, 800],
            [0.65, 900],
            [0.75, 100], [0.75, 200], [0.75, 300], [0.75, 400], [0.75, 500], [0.75, 600], [0.75, 700], [0.75, 800],
            [0.75, 900],
            [0.85, 100], [0.85, 200], [0.85, 300], [0.85, 400], [0.85, 500], [0.85, 600], [0.85, 700], [0.85, 800],
            [0.85, 900],
            [0.95, 100], [0.95, 200], [0.95, 300], [0.95, 400], [0.95, 500], [0.95, 600], [0.95, 700], [0.95, 800],
            [0.95, 900],

            ])

    # [0.25, 550], [0.25, 600], [0.55, 550], [0.55, 600], [0.85, 550], [0.85, 600]
    #[0.25,200],[0.25,400],[0.25,800],[0.55,200],[0.55,400],[0.55,800],[0.85,200],[0.85,400],[0.85,800]

    Num_sim_data_variable = num_sim_data(sampling_type='variable', sampling_num=config.sampling_number,
                                      sparsetocompact=config.sparsetocompact, switch_print=config.print_index,
                                      impact_window=config.impact_window, train_path=config.origin_train_data,
                                      test_path=config.origin_test_data, test_mononum=config.singlefreq_test_number)
    x_train_val_variable, y_train_val_variable = Num_sim_data_variable.triandata(swit_printindex=True)
    x_test_variable, y_test_variable = Num_sim_data_variable.testdata(
        ampfreqs=[

            [0.25, 100], [0.25, 200], [0.25, 300], [0.25, 400], [0.25, 500], [0.25, 600], [0.25, 700], [0.25, 800],
            [0.25, 900],
            [0.35, 100], [0.35, 200], [0.35, 300], [0.35, 400], [0.35, 500], [0.35, 600], [0.35, 700], [0.35, 800],
            [0.35, 900],
            [0.45, 100], [0.45, 200], [0.45, 300], [0.45, 400], [0.45, 500], [0.45, 600], [0.45, 700], [0.45, 800],
            [0.45, 900],
            [0.55, 100], [0.55, 200], [0.55, 300], [0.55, 400], [0.55, 500], [0.55, 600], [0.55, 700], [0.55, 800],
            [0.55, 900],
            [0.65, 100], [0.65, 200], [0.65, 300], [0.65, 400], [0.65, 500], [0.65, 600], [0.65, 700], [0.65, 800],
            [0.65, 900],
            [0.75, 100], [0.75, 200], [0.75, 300], [0.75, 400], [0.75, 500], [0.75, 600], [0.75, 700], [0.75, 800],
            [0.75, 900],
            [0.85, 100], [0.85, 200], [0.85, 300], [0.85, 400], [0.85, 500], [0.85, 600], [0.85, 700], [0.85, 800],
            [0.85, 900],
            [0.95, 100], [0.95, 200], [0.95, 300], [0.95, 400], [0.95, 500], [0.95, 600], [0.95, 700], [0.95, 800],
            [0.95, 900],

        ])

    x_train_val=np.concatenate((x_train_val_equal,x_train_val_variable),axis=2)
    x_test=np.concatenate((x_test_equal,x_test_variable),axis=2)
    shuffle_index_train = np.random.permutation(np.arange(len(x_train_val)))
    shuffle_index_test = np.random.permutation(np.arange(len(x_test)))
    x_train_val=x_train_val[shuffle_index_train]
    y_train_val=y_train_val_equal[shuffle_index_train]
    x_test=x_test[shuffle_index_test]
    y_test=y_test_equal[shuffle_index_test]

    num_train=int(len(x_train_val)*config.train_ratio)
    x_train=x_train_val[:num_train]
    y_train=y_train_val[:num_train]
    x_valitation=x_train_val[num_train:]
    y_valitation=y_train_val[num_train:]
    print('-----------------------------------------')
    print('Number origin_train_standard:',len(x_train))
    print('Number valid:', len(x_valitation))
    print('Number origin_test_standard:', len(x_test))
    print('-----------------------------------------')

    return  x_train,y_train,x_valitation,y_valitation,x_test,y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_train_data', type=str, default='./dataset/train_inlet_velocity_0.6/')
    parser.add_argument('--origin_test_data', type=str, default='./dataset/test_inlet_velocity_0.6/')
    parser.add_argument('--sampling_num', type=int, default=1000)
    parser.add_argument('--sparsetocompact', type=bool, default=True)
    parser.add_argument('--switch_print', type=bool, default=True)
    parser.add_argument('--impact_window', type=int, default=6000)
    parser.add_argument('--index_maxboundary', type=float, default=6000)
    parser.add_argument('--singlefreq_nums', type=int, default=1200)
    parser.add_argument('--train_ratio', type=float, default=0.98)
    config = parser.parse_args()
    x_train,y_train,x_valitation,y_valitation,x_test,y_test=Dataset(config)
    print(x_train.shape,y_train.shape,x_valitation.shape,y_valitation.shape,x_test.shape,y_test.shape)
