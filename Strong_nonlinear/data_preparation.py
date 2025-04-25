import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch.utils import data


class Interval_sampling:
    def __init__(self,num,switch_print,switch_plot,max_length):
        self.num=num
        self.switch_print=switch_print
        self.switch_plot=switch_plot
        self.max_length=max_length

    def variable_interval(self,reversal):
        sampling_num = np.arange(1, self.num + 1, 1)
        sum_sampling_num = np.sum(sampling_num)
        delta = self.max_length / sum_sampling_num

        if reversal == True:
            a = self.max_length
        else:
            a = 0
        dn = 0
        index_list = []
        count = []
        for i in range(self.num):
            dn = dn + delta
            if reversal == True:
                a = a - dn
            else:
                a = a + dn
            index_list.append(int(a))
            count.append(i)
        index_list = np.array(index_list)
        if reversal == True:
            index_list = index_list[::-1]
        if self.switch_print == True:
            print('采样索引信息------------------------------------')
            print('sampling:variable')
            if reversal==True:
                print('reversal:True')
            else:
                print('reversal:False')
            count = np.array(count)
            print('dn:', dn)
            print('a:', a)
            print('index_list:', index_list.shape, '\n', 'count:', count.shape, sep='')
            print('---------------------------------------------')
        if self.switch_plot == True:
            plt.figure(0)
            plt.scatter(index_list, count, s=5)
            plt.tick_params(labelsize=27)
            plt.show()
        return index_list

    def equal_interval(self,):
        index_list=np.linspace(0,self.max_length,self.num,dtype=int)
        if self.switch_print == True:
            print('采样索引信息------------------------------------')
            print('sampling:equal')
            print('index_list:', index_list.shape, sep='')
            print('---------------------------------------------')
        return index_list

class Numerical_sim_data:

    def __init__(self, switch_sampling, num, reversal, swit_print, swit_plot, maxlen,patch_seq):
        self.switch_sampling = switch_sampling
        self.num = num
        self.maxlen = maxlen
        self.reversal = reversal
        self.swit_print = swit_print
        self.swit_plot = swit_plot
        self.patch_seq=patch_seq

    def triandata(self, swit_printindex,sweepornoise):
        global x_total, y_total
        x_index=None
        if self.switch_sampling != False:
            sampling = Interval_sampling(num=self.num, max_length=self.maxlen, switch_print=self.swit_print,
                                         switch_plot=self.swit_plot)
            if self.switch_sampling == 'variable':
                x_index = sampling.variable_interval(reversal=self.reversal)
            else:
                x_index = sampling.equal_interval()
            if swit_printindex==True:
                print('采样索引：',x_index)
        if sweepornoise=='sweep':
            for i in range(1, 11, 1):
                path = r'D:\wjw\FDF_数值模拟数据\sweep_0.05s_10Hz\amp' + str(i / 10) + '.csv'
                print(path)
                data_df = pd.read_csv(path, header=0)
                x_y = data_df.to_numpy()
                x = x_y[:, 0]
                y = x_y[:, 1]
                x = np.lib.stride_tricks.sliding_window_view(x[1:], self.patch_seq, axis=0)
                y = y[self.patch_seq:]
                if self.switch_sampling != False:
                    x = x[:, x_index]
                if i == 1:
                    x_total = x
                    y_total = y
                    print('x:', x.shape, x.dtype, type(x), '\n', 'y:', y.shape, sep='')
                    continue
                print('x:', x.shape, x.dtype, type(x), '\n', 'y:', y.shape)
                x_total = np.concatenate((x_total, x), axis=0)
                y_total = np.concatenate((y_total, y), axis=0)
        else:
            path=r'D:\wjw\FDF_数值模拟数据\sweep_0.05s_10Hz\noise.csv'
            print(path)
            data_df=pd.read_csv(path,header=0)
            x_y=data_df.to_numpy()
            x = x_y[:, 0]
            y = x_y[:, 1]
            x = np.lib.stride_tricks.sliding_window_view(x[1:], self.patch_seq, axis=0)
            y = y[self.patch_seq:]
            if self.switch_sampling != False:
                x = x[:, x_index]
            x_total=x
            y_total=y

        x_total = x_total.astype(np.float32)
        y_total = y_total.astype(np.float32)
        x_total = x_total.reshape((x_total.shape[0], x_total.shape[1], 1))
        y_total = y_total.reshape((y_total.shape[0], 1))
        x_total = torch.from_numpy(x_total)
        y_total = torch.from_numpy(y_total)
        print('x_total:', x_total.shape, type(x_total), '\n', 'y_total:', y_total.shape, type(y_total), '\n',
              'reversal:', self.reversal, sep='')
        return x_total, y_total

    def testdata_monofrequcy(self, path):
        data_df = pd.read_csv(path, header=0)
        x_y = data_df.to_numpy()
        x = x_y[:, 0]
        y = x_y[:, 1]
        x = np.lib.stride_tricks.sliding_window_view(x[1:], self.patch_seq, axis=0)
        y = y[self.patch_seq:]
        if self.switch_sampling != False:
            sampling = Interval_sampling(num=self.num, max_length=self.maxlen, switch_print=self.swit_print,
                                         switch_plot=self.swit_plot)
            if self.switch_sampling == 'variable':
                x_index = sampling.variable_interval(reversal=self.reversal)
                x = x[:, x_index]
            else:
                x_index = sampling.equal_interval()
                x = x[:, x_index]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x = x.reshape((x.shape[0], x.shape[1], 1))
        y = y.reshape((y.shape[0], 1))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x[-512:].cuda()
        y = y[-512:].cuda()
        print('单频测试集:', x.shape, '\n', 'reversal:', self.reversal, sep='')
        return x, y

def white_noise_distur_change():
    a=np.random.normal(loc=0.0, scale=1.0, size=500000)
    a_cdf=norm.cdf(a,0,1)
    def inv_q(x):
        a=(x-0.5)*2*1
        return a
    b=inv_q(a_cdf)*3.5+1
    b_fft=np.fft.fft(b)
    a_fft=np.fft.fft(a)
    c=np.random.uniform(-1,1,1000)
    c_fft=np.fft.fft(c)

    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure()
    # plt.plot(np.abs(a_fft))
    # plt.title('白噪声fft')
    # plt.figure()
    # plt.plot(np.abs(b_fft))
    # plt.title('白噪声均匀分布投影fft')
    # #plt.figure()
    # #plt.plot(np.abs(c_fft))
    # #plt.title('均匀分布fft')
    # plt.figure()
    # plt.hist(a,bins=50)
    # plt.title('白噪声幅值柱状图')
    # plt.figure()
    # plt.hist(b,bins=50)
    # plt.title('白噪声幅值均匀分布投影柱状图')
    # b_df=pd.DataFrame(b)
    # b_df.to_csv('F://brandband.csv')
    plt.figure()
    plt.plot(b,color=(17/255,50/255,93/255))
    #plt.axis('off')
    plt.show()
    return b

def nlds_datapre(path,seq_len):
    print(path)
    data_df = pd.read_csv(path)
    x_y = data_df.to_numpy()
    x = x_y[:, 0]
    y = x_y[:, 1]
    x = np.lib.stride_tricks.sliding_window_view(x[1:], seq_len, axis=0)
    y = y[seq_len:]
    # x_index = np.arange(0, 8000, 5, dtype=int)
    # x = x[:, x_index]
    x=x.astype(np.float32)
    y=y.astype(np.float32)
    x=x.reshape((x.shape[0],x.shape[1],1))
    y=y.reshape((y.shape[0],1))
    x=torch.from_numpy(x)
    y=torch.from_numpy(y)
    return x,y

def nlds_testdata_monofrequcy(seq_len,a1,a3,):

    amps=[0.1,0.4,0.5,0.25,0.45]
    freqs=[100,350,600,850]
    x_tests=[]
    y_tests=[]
    for amp in amps:
        for freq in freqs:
            path_mono = r'./Different_datasize_and_nonlinearity\omegac400_tau1_tau3_2e-3\testdata\a1_' + str(a1) + '_a3_' + str(a3) + '_' + str(
                amp) + '_' + str(freq) + '.csv'
            data_df = pd.read_csv(path_mono, header=0)
            x_y = data_df.to_numpy()
            x = x_y[:, 0]
            y = x_y[:, 1]
            x = np.lib.stride_tricks.sliding_window_view(x[1:], seq_len, axis=0)
            y = y[seq_len:]
            x=x.astype(np.float32)
            y=y.astype(np.float32)
            x = x.reshape((x.shape[0], x.shape[1], 1))
            y = y.reshape((y.shape[0], 1))

            sampling_index = np.linspace(0, len(x) - 1, 500, dtype=int)
            x=x[sampling_index]
            y=y[sampling_index]

            x_tests.append(x)
            y_tests.append(y)

    x_tests=np.concatenate(x_tests,axis=0)
    y_tests=np.concatenate(y_tests,axis=0)

    x = torch.from_numpy(x_tests)
    y = torch.from_numpy(y_tests)
    print('单频测试集:', x.shape)

    dataset_train = data.TensorDataset(x, y)

    Dataloaders_test = data.DataLoader(dataset=dataset_train,
                                        batch_size=256,
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True)

    return Dataloaders_test



if __name__ == '__main__':
    v_ini=np.ones((int(0.05/1e-6),1))
    v_x=white_noise_distur_change()
    v_x=v_x.reshape(v_x.shape[0],1)
    v_x=np.concatenate((v_ini,v_x),axis=0)
    n = v_x.shape[0]
    v_y_z=np.zeros((n,2))
    t=np.arange(0,1e-6*n,1e-6).reshape(n,1)
    data=np.concatenate((t,v_x,v_y_z),axis=1)
    print(data.shape)
    data=pd.DataFrame(data)
    data.columns=['time','velocity-x','velocity-y','velocity-z']
    data.to_csv(r'D:\CFD\blueCFD\blueCFD-Core-2017\ofuser-of5\run\case1\data.csv',index=False)


