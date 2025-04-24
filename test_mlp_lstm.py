import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from network import model_mlp, model_lstm
from torch.utils import data
import tqdm

class num_sim_test:

    def __init__(self,impact_window):

        self.impact_window=impact_window

    def plot_single(self, model_type ,model_path, ampfreq, plot_seq_len):
        device = torch.device('cpu')
        print('model_path:', model_path)
        model= model_type
        model.load_state_dict(torch.load(model_path,map_location=device))
        model.eval()
        # model = torch.load(model_path, map_location=device)
        # -----------------------------------------------------------------
        # plt.figure()
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        dt = 1e-6
        fig, axarr = plt.subplots(1, sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0})
        data_path = r'dataset\origin_test_meanv1\\' + str(ampfreq[0]) + '_' + str(ampfreq[1]) + 'Hz.csv'
        data_df = pd.read_csv(data_path)
        data_np = data_df.to_numpy(dtype=np.float32)
        print(data_np.shape, type(data_np), data_np.dtype)
        dataclear_x = data_np[plot_seq_len:, 0]
        dataclear_y = data_np[plot_seq_len:, 1]
        x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
        y = dataclear_y[self.impact_window:]
        # x=torch.from_numpy(x)
        x = x.reshape((x.shape[0], x.shape[1], 1))
        print('x:', x.shape, '\n', 'y:', y.shape, sep='')

        y = y / 0.1792  # 0.1792 #0.1078
        with torch.no_grad():
            x_input = torch.tensor(x)
            y_pre1 = model(x_input)
            y_pre1 = torch.squeeze(y_pre1) / 0.1792
            error1 = torch.mean(torch.abs(y_pre1 - y))
            mean = np.mean(np.abs(y))
            re_error1 = error1 / mean
            print("MRE：", re_error1)
        x = np.linspace(0, len(y) * dt, len(y))
        axarr.plot(x, y, label='Numerical simulation', linewidth=8, color=(245 / 255, 166 / 255, 115 / 255),
                   alpha=0.5)
        axarr.plot(x, y_pre1, label='Neural network', linewidth=4,
                   color=(17 / 255, 50 / 255, 93 / 255), linestyle='--')
        xmin = min(x)
        xmax = max(x)
        ymin = min(min(y), min(y_pre1)) * 1.1
        ymax = max(max(y), max(y_pre1)) * 1.1
        ymax_abs = max(abs(ymin), abs(ymax))
        axarr.tick_params(labelsize=25, direction='in', length=6, width=2, top=True, right=True)
        axarr.set_xlim(xmin, xmax)
        axarr.set_ylim(-ymax_abs, ymax_abs)
        axarr.spines['bottom'].set_linewidth(2)
        axarr.spines['left'].set_linewidth(2)
        axarr.spines['top'].set_linewidth(2)
        axarr.spines['right'].set_linewidth(2)
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc=[0.29, 0.89], fontsize=10, ncol=3, frameon=False,
                   prop={'family': 'Times New Roman', 'size': 25})

        font1 = {'family': 'Times New Roman', 'size': 30}
        font2 = {'family': 'Times New Roman', 'size': 30}
        fig.text(0.5, 0.03, r'$t/s$', va='center', fontdict=font1, usetex=True)
        fig.text(0.04, 0.5, r"$q^{'}/ \overline{q}$", va='center', rotation='vertical', fontdict=font2, usetex=True)
        plt.show()

    def plot_multi(self,model_type,model_path,ampfreqs,plot_seq_len):
        device = torch.device('cpu')
        print('model_path:', model_path)
        model = model_type
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        # model = torch.load(model_path, map_location=device)

        # -----------------------------------------------------------------
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        dt = 1e-6
        ampfreqs_count = len(ampfreqs)
        fig, axarr = plt.subplots(ampfreqs_count, sharex='col', gridspec_kw={'hspace': 0.1, 'wspace': 0})
        letter_num = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)',4: '(e)'}
        for i, ampfreq in enumerate(ampfreqs):
            data_path = r'dataset\origin_test_meanv1\\' + str(ampfreq[0]) + '_' + str(ampfreq[1]) + 'Hz.csv'
            data_df = pd.read_csv(data_path)
            data_np = data_df.to_numpy(dtype=np.float32)
            print(data_np.shape, type(data_np), data_np.dtype)
            dataclear_x = data_np[plot_seq_len:, 0]
            dataclear_y = data_np[plot_seq_len:, 1]
            x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
            y = dataclear_y[self.impact_window:]
            x = x.reshape((x.shape[0], x.shape[1], 1))
            print('x:', x.shape, '\n', 'y:', y.shape, sep='')


            plot_num = int((2 / ampfreq[1]) / dt)
            x = x[:plot_num]
            y = y[:plot_num]
            y = y / 0.1792 / ampfreq[0]  # 0.1792 #0.1078

            # with torch.no_grad():
            #     x_input = torch.tensor(x_input)
            #     y_pre = model(x_input)
            #     y_pre = torch.squeeze(y_pre1) / 0.1792 / ampfreq[0]
            #     error = torch.mean(torch.abs(y_pre - y))
            #     mean = np.mean(np.abs(y))
            #     re_error1 = error / mean
            #     print("MRE：", re_error1)


            x = torch.tensor(x)
            y = torch.tensor(y)
            dataset = data.TensorDataset(x, y)
            data_loader = data.DataLoader(dataset=dataset,
                                          batch_size=128,
                                          shuffle=False,
                                          drop_last=False)
            y_pre_total = []
            y_total = []
            with torch.no_grad():
                for x, y in tqdm.tqdm(data_loader):
                    y_pre = model(x)
                    y_pre = torch.squeeze(y_pre)/ 0.1792 / ampfreq[0]
                    y_total.append(y)
                    y_pre_total.append(y_pre)

            y = torch.cat(y_total)
            y_pre = torch.cat(y_pre_total)


            error = torch.mean(torch.abs(y_pre - y))
            mean = torch.mean(torch.abs(y))
            re_error = error / mean
            print("Average MRE：", re_error)


            x = np.linspace(0, len(y) * dt / (1 / ampfreq[1]), len(y))
            axarr[i].plot(x, y, label='Numerical simulation', linewidth=8, color=(245 / 255, 166 / 255, 115 / 255),
                          alpha=0.5)
            axarr[i].plot(x, y_pre, label='Neural network', linewidth=4,
                          color=(17 / 255, 50 / 255, 93 / 255), linestyle='--')
            xmin = min(x)
            xmax = max(x)
            ymin = min(min(y), min(y_pre)) * 1.1
            ymax = max(max(y), max(y_pre)) * 1.1
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

        font1 = {'family': 'Times New Roman', 'size': 35}
        font2 = {'family': 'Times New Roman', 'size': 35}
        fig.text(0.5, 0.03, r'$t/T$', va='center', fontdict=font1, usetex=True)
        fig.text(0.04, 0.5, r"$q^{'}/ \overline{q} /A$", va='center', rotation='vertical', fontdict=font2, usetex=True)

        plt.show()

    def plot_ampfreqs_models(self,model_path,ampfreqs,model_num,plot_seq_len):
        device = torch.device('cpu')
        print('model_path:', model_path)
        model1 = torch.load(model_path[0], map_location=device)
        # model2 = torch.load(model_path[1],map_location=device)
        # model1 = model_transformer()
        # model1.load_state_dict(torch.load(model_path[0], map_location=device))
        #
        model2 = model_transformer_no_posi()
        model2.load_state_dict(torch.load(model_path[1], map_location=device))

        # model1.eval()
        model2.eval()
        # -----------------------------------------------------------------
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.rc('font', family='Times New Roman')
        dt = 1e-6
        ampfreqs_count = len(ampfreqs)
        fig, axarr = plt.subplots(ampfreqs_count, sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0})
        y_total=[]
        y_pre1_total=[]
        y_pre2_total=[]
        letter_num={0:'(a)',1:'(b)',2:'(c)',3:'(d)'}
        for i, ampfreq in enumerate(ampfreqs):

            data_path = r'.\dataset\origin_test_meanv1\\' + str(ampfreq[0]) + '_' + str(ampfreq[1]) + 'Hz.csv'
            data_df = pd.read_csv(data_path)
            data_np = data_df.to_numpy(dtype=np.float32)
            print(data_np.shape, type(data_np), data_np.dtype)
            dataclear_x = data_np[plot_seq_len:, 0]
            dataclear_y = data_np[plot_seq_len:, 1]
            x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
            y = dataclear_y[self.impact_window:]
            # x=torch.from_numpy(x)
            x = torch.tensor(x)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            print('x:', x.shape, '\n', 'y:', y.shape, sep='')
            Sampling = sampling(num=model_num, max_length=self.index_maxboundary, switch_print=self.swit_print)
            if self.sampling_type == 'variable':
                x_index = Sampling.variable_interval(sparsetocompact=self.sparsetocompact)
                x_index=x_index.copy()
                x_input = x[:, x_index]
            else:
                x_index = Sampling.equal_interval()
                x_input = x[:, x_index]

            plot_num = int((2 / ampfreq[1]) / dt)
            x_input=x_input[:plot_num]
            y=y[:plot_num]
            y=y / 0.1792 / ampfreq[0]
            with torch.no_grad():
                y_pre1 = model1(x_input)
                y_pre1 = torch.squeeze(y_pre1)/0.1792/ampfreq[0]
                error1 = torch.mean(torch.abs(y_pre1 - y))
                mean = np.mean(np.abs(y))
                re_error1 = error1 / mean
                print("相对误差1：", re_error1)

                y_pre2 = model2(x_input[:plot_num])
                y_pre2 = torch.squeeze(y_pre2)/0.1792/ampfreq[0]
                error2 = torch.mean(torch.abs(y_pre2 - y))
                re_error2 = error2 / mean
                print("相对误差2：", re_error2)
            x=np.linspace(0, len(y) * dt/(1/ampfreq[1]), len(y))
            axarr[i].plot(x, y, label='Numerical simulation', linewidth=8, color=(245 / 255, 166 / 255, 115 / 255),
                          alpha=0.5)
            axarr[i].plot(x, y_pre1, label='Chronological', linewidth=4,
                          color=(17 / 255, 50 / 255, 93 / 255))
            axarr[i].plot(x, y_pre2, label='No chronological', linestyle='--',
                          linewidth=5,
                          color=(115 / 255, 107 / 255, 157 / 255))
            xmin=min(x)
            xmax = max(x)
            ymin=min(min(y),min(y_pre1),min(y_pre2))*1.2
            ymax=max(max(y),max(y_pre1),max(y_pre2))*1.2
            ymax_abs=max(abs(ymin),abs(ymax))
            print(ymin,ymax)
            axarr[i].tick_params(labelsize=25,direction='in',length=6,width=2,top=True,right=True)
            axarr[i].text(0.95*xmax,0.6*ymax_abs,letter_num[i],fontsize=25)
            axarr[i].set_xlim(xmin,xmax)
            axarr[i].set_ylim(-ymax_abs,ymax_abs)
            axarr[i].set_xticks([0,0.5,1,1.5,2])
            axarr[i].spines['bottom'].set_linewidth(2)
            axarr[i].spines['left'].set_linewidth(2)
            axarr[i].spines['top'].set_linewidth(2)
            axarr[i].spines['right'].set_linewidth(2)

            # y=y[:,np.newaxis]
            # y_total.append(y)
            # y_pre1_total.append(y_pre1.unsqueeze(dim=1))
            # y_pre2_total.append(y_pre2.unsqueeze(dim=1))


        # y_total=np.concatenate(y_total,axis=1)
        # y_pre1_total=np.concatenate(y_pre1_total,axis=1)
        # y_pre2_total=np.concatenate(y_pre2_total,axis=1)
        # y_df=pd.DataFrame(y_total)
        # y_pre1_df=pd.DataFrame(y_pre1_total)
        # y_pre2_df=pd.DataFrame(y_pre2_total)
        # y_df.to_csv(r'D:\wjw\FDF_pythonproject\y.csv', header=['200Hz', '400Hz', '600Hz', '800Hz'])
        # y_pre1_df.to_csv(r'D:\wjw\FDF_pythonproject\y1.csv',header=['200Hz','400Hz','600Hz','800Hz'])
        # y_pre2_df.to_csv(r'D:\wjw\FDF_pythonproject\y2.csv',header=['200Hz','400Hz','600Hz','800Hz'])
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc=[0.2,0.89], fontsize=10, ncol=3, frameon=False,prop={'family': 'Times New Roman','size':25})

        font1 = {'family': 'Times New Roman', 'size': 30}
        font2 = {'family': 'Times New Roman', 'size': 30}
        fig.text(0.5, 0.03, r'$t/T$',  va='center',fontdict=font1,usetex=True)
        fig.text(0.03, 0.5, r"$q^{'}/ \overline{q} /A$", va='center', rotation='vertical',fontdict=font2,usetex=True)

        plt.show()

    def plot_MRE_statistics(self,model_type,model_path):
        device = torch.device('cpu')
        print('model_path:', model_path)
        model = model_type
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        # model = torch.load(model_path, map_location=device)
        # -----------------------------------------------------------------
        # plt.rcParams['axes.unicode_minus'] = False
        # ----------------------statistical relative error------------------------------------------------------------
        x_multifreqs = []
        y_multifreqs = []
        amps=np.arange(0.25,0.96,0.1)
        freqs=np.arange(100,901,100)
        for amp in amps:
            for freq in freqs:
                data_path = r'dataset\origin_test_meanv1\\' + str(amp) + '_' + str(freq) + 'Hz.csv'
                data_df = pd.read_csv(data_path, header=0)
                x_y = data_df.to_numpy()
                x = x_y[:, 0]
                y = x_y[:, 1]
                x = np.lib.stride_tricks.sliding_window_view(x[1:], self.impact_window, axis=0)
                x = x.reshape((x.shape[0], x.shape[1], 1))
                y = y[self.impact_window:]
                sampling_index = np.linspace(0, len(x) - 1, 1200, dtype=int)
                x_multifreqs.append(x[sampling_index, :])
                y_multifreqs.append(y[sampling_index])

        x = np.concatenate(x_multifreqs, axis=0).astype(np.float32)
        y = np.concatenate(y_multifreqs, axis=0).astype(np.float32)


        x = torch.tensor(x)
        y = torch.tensor(y)
        dataset = data.TensorDataset(x, y)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=128,
                                      shuffle=False,
                                      drop_last=False)
        y_pre_total = []
        y_total = []
        with torch.no_grad():
            for x, y in data_loader:
                y_pre, _ = model(x)
                y_pre = torch.squeeze(y_pre)
                y_total.append(y)
                y_pre_total.append(y_pre)

        y_total = torch.cat(y_total)
        y_pre_total = torch.cat(y_pre_total)
        print(y_total.shape)

        error = torch.mean(torch.abs(y_pre_total - y_total))
        mean = torch.mean(torch.abs(y_total))
        re_error = error / mean
        print("Average MRE：", re_error)


if __name__ == '__main__':
     model_type = model_lstm()
     model_path1 = r'E:\Paper_two\FDF_pythonproject\The_Other_Model\checkpoint(mlp)\model-mlp--6000--14.2%(last_epoch).pth'
     model_path2 = r'E:\Paper_two\FDF_pythonproject\The_Other_Model\checkpoint(lstm)\model-lstm--6000--3.5%(best).pth'
     testplot=num_sim_testplot(impact_window=6000)
     s=time.time()
     testplot.plot_multi(model_type=model_type,model_path=model_path2,ampfreqs=[[0.85,200],[0.85,400],[0.85,600],[0.85,800]],plot_seq_len=-16000)
     # testplot.plot_MRE_statistics(model_type=model_type,model_path=model_path1)
     # testplot.plot_ampfreqs_models(model_path=model_path,ampfreqs=[[0.55,200],[0.55,300],[0.55,400],[0.55,800]],model_num=1000,plot_seq_len=-16000)

     e=time.time()
     print('时间：',e-s)