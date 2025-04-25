import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FuncFormatter

from data_preparation import nlds_datapre

def plot_test(y,y_pre):
    dt = 1e-5
    t = np.arange(0, dt * (len(y)), dt)
    print('t:', t.shape)
    plt.figure(0)
    plt.plot(t, y, label='y')
    plt.plot(t, y_pre, label='y_pre')
    plt.show()

def mono_test(path_data,path_model,seqlen):
    divice=torch.device('cpu')
    model=torch.load(path_model,map_location=divice)
    x,y=nlds_datapre(path_data,seq_len=seqlen)
    x=x[-2000:]
    y=y[-2000:]
    #print('x:',x.shape,'\n','y:',y.shape,sep='')
    dt=1*10e-5
    plot_num = int((2 / 300) / dt)
    x = x[:plot_num]
    y = y[:plot_num]
    y = y /0.35

    with torch.no_grad():
        y_pre=model(x)/0.35
        #print('y_pre:',y_pre.shape)
    error = torch.mean(torch.abs(y_pre - y))
    mean = torch.mean(torch.abs(y))
    re_error = error / mean
    print("相对误差：", re_error.detach().cpu().numpy())
    #plot_test(y,y_pre)
    return y,y_pre,re_error

def multiintensity_test(size,seq_len):
    global y_total, y_pre_total
    intensities=range(1,6,1)

    for intensity in intensities:
        path_data = r'./Different_datasize_and_nonlinearity\omegac400_tau1_tau3_2e-3\testdata\a1_1_a3_'+str(intensity)+'_0.5_350.csv'
        path_model='checkpoint_nonlinear_omegac400_tau1_tau3_2e-3\model_cnn_lstm_a1_1_a3_'+str(intensity)+'_'+str(size)+'.pth'


        y,y_pre,re_error=mono_test(path_data,path_model,seqlen=seq_len)
        if intensity==1:
            y_total=y
            y_pre_total=y_pre
            print('y:', y_total.shape, '\n', 'y_pre:', y_pre_total.shape, sep='')
        else:
            y_total=np.concatenate((y_total,y),axis=1)
            y_pre_total=np.concatenate((y_pre_total,y_pre),axis=1)
            print('y:', y_total.shape, '\n', 'y_pre:', y_pre_total.shape, sep='')

    dt = 1e-5
    t = np.arange(0, dt * (len(y_total)), dt)
    ampfreqs_count = len(intensities)
    fig, axarr = plt.subplots(ampfreqs_count, sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0})
    letter_num = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)'}
    for i in range(len(intensities)):
        y=y_total[:, i]
        y_pre=y_pre_total[:,i]
        x = np.linspace(0, len(y) * dt / (1 / 300), len(y))
        axarr[i].plot(x,y, label='Numerical simulation', linewidth=7, color=(245 / 255, 166 / 255, 115 / 255),
                      alpha=0.5)
        axarr[i].plot(x, y_pre, label='Neural network', linewidth=3.5,
                      color=(17 / 255, 50 / 255, 93 / 255), linestyle='--')
        xmin = min(x)
        xmax = max(x)
        ymin = min(min(y), min(y_pre)) * 1.2
        ymax = max(max(y), max(y_pre)) * 1.2
        ymax_abs = max(abs(ymin), abs(ymax))
        axarr[i].tick_params(labelsize=25, direction='in', length=6, width=2, top=True, right=True)
        axarr[i].text(0.95 * xmax, 0.6 * ymax_abs, letter_num[i], fontsize=25)
        axarr[i].set_xlim(xmin, xmax)
        axarr[i].set_ylim(-ymax_abs, ymax_abs)
        axarr[i].set_xticks([0, 0.5, 1, 1.5, 2])
        axarr[i].spines['bottom'].set_linewidth(2)
        axarr[i].spines['left'].set_linewidth(2)
        axarr[i].spines['top'].set_linewidth(2)
        axarr[i].spines['right'].set_linewidth(2)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc=[0.29, 0.89], fontsize=10, ncol=3, frameon=False,
               prop={'family': 'Times New Roman', 'size': 25})
    font1 = {'family': 'Times New Roman', 'size': 30}
    font2 = {'family': 'Times New Roman', 'size': 30}
    fig.text(0.5, 0.03, r'$t/T$', va='center', fontdict=font1, usetex=True)
    fig.text(0.03, 0.5, r"$q^{'}/ \overline{q} /A$", va='center', rotation='vertical', fontdict=font2, usetex=True)
    plt.show()

def Precision_map(amp,freq,omegac):
    intensties=[4,8,16,25,32]
    sizes=range(20000,100001,20000)
    re_error_total=[]
    for size in sizes:
        re_error_size=[]
        for intensity in intensties:
            path_data = r'F:\数据集长度与非线性强度关系\omegac'+str(omegac)+r'\test_omegac'+str(omegac)+r'\a1_1_a3_' + str(intensity) + '_'+str(amp)+'_'+str(freq)+'.csv'
            path_model = 'checkpoint_nonlinear_omegac500\model_cnn_lstm_a1_1_a3_' + str(intensity) + '_' + str(size) + '.pth'
            print(path_model)
            y,y_pre,re_error=mono_test(path_data,path_model)
            re_error_size.append(re_error)

        re_error_size=np.array(re_error_size)
        print('re_error_size:',re_error_size)
        if size==20000:
            re_error_total=re_error_size.reshape(re_error_size.shape[0],1)
        else:
            re_error_total=np.append(re_error_total,re_error_size.reshape(re_error_size.shape[0],1),axis=1)
            print('re_error:',re_error_total.shape,'\n',re_error_total,sep='')
    plt.figure(str(amp)+'---'+str(freq)+'---omegac'+str(omegac))
    for i,intensity in enumerate(intensties):
        plt.plot(sizes,re_error_total[i,:]*100,label=str(intensity),marker='o',markersize=6,linewidth=3)
    plt.legend()
    plt.title(str(amp)+'---'+str(freq)+'---omegac'+str(omegac))
    def to_percent(x,position):
        return '%1.1f' % (x) + '%'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()

class Nl_Ds_test:
    def __init__(self,seqlen):
        self.seq_len=seqlen
    def re_error_stastics(self,amps_freqs,intensties,sizes):
        # intensties = [4, 8, 16, 25, 32]
        # sizes = range(20000, 100001, 20000)
        colors=[(245/255,166/255,115/255),(183/255,131/255,175/255),(115/255,107/255,157/255),
                (54/255,80/255,131/255),(17/255,50/255,93/255)]
        re_error_total_add=0
        for amp_freq in amps_freqs:
            re_error_total = []
            for size in sizes:
                re_error_size = []
                for intensity in intensties:
                    path_data = r'D:\wjw\数据集长度与非线性强度关系\omegac400_tau1_tau3_2e-3\testdata\a1_1_a3_'+str(intensity)+'_'+str(amp_freq[0])+'_'+str(amp_freq[1])+'.csv'
                    path_model = 'checkpoint_nonlinear_omegac400_tau1_tau3_2e-3\model_cnn_lstm_a1_1_a3_' + str(intensity) + '_' + str(size) + '_new.pth'
                    print(path_model)
                    y, y_pre, re_error = mono_test(path_data, path_model,seqlen=self.seq_len)
                    re_error_size.append(re_error)

                re_error_size = np.array(re_error_size)
                if size == 20000:
                    re_error_total = re_error_size.reshape(re_error_size.shape[0], 1)
                else:
                    re_error_total = np.append(re_error_total, re_error_size.reshape(re_error_size.shape[0], 1), axis=1)
                    print('re_error_total:', re_error_total.shape, '\n', re_error_total, sep='')
            re_error_total_add=re_error_total_add+re_error_total
        print('re_error_total_add:',re_error_total_add)
        re_error_total_mean=re_error_total_add/len(amps_freqs)
        print('len(amps_freqs):',len(amps_freqs),'\n','re_error_total_mean:',re_error_total_mean,sep='')
        plt.figure('average')
        for i, intensity in enumerate(intensties):
            plt.plot(sizes, re_error_total_mean[i, :] * 100, label='$a_{3}/a_{1}$='+str(intensity), marker='o', markersize=10, linewidth=5,color=colors[i])
        plt.legend(fontsize='28')
        plt.tick_params(labelsize=24)
        plt.xlabel('Training dataset sample size', fontsize=24)
        plt.ylabel('Mean relative error', fontsize=24)
        #plt.title('average')
        def to_percent(x, position):
            return '%1.1f' % (x) + '%'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        # plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.show()

if __name__ == '__main__':
    # test=Nl_Ds_test(seqlen=225)
    # a = []
    # for amp in [0.5]:
    #     for freq in np.arange(100, 1001, 100):
    #         a.append([round(amp,3), int(freq)])
    # test.re_error_stastics(a,[1,2,3,4,5],range(20000,100001,20000))
    multiintensity_test(100000,seq_len=225)





