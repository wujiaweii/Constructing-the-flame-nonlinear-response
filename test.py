import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from network import model_dual_path,model_single_path,model_lstm,model_mlp,model_dual_path_nonlinear
from torch.utils import data
import tqdm
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
            print('Information of sampling index------------------------------------')
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
            print('Information of sampling index------------------------------------')
            print('sampling:equal')
            print('index_list:', index_list.shape, sep='')
            print('---------------------------------------------')
        return index_list

class num_sim_testplot:

    def __init__(self,print_index,impact_window,sample_num):

        self.print_index = print_index
        self.impact_window = impact_window
        self.sample_num = sample_num

    def plot_multi(self, model_type, model_path, ampfreqs, plot_seq_len):
        device = torch.device('cpu')
        print('model_path:', model_path)
        model=model_type
        model.load_state_dict(torch.load(model_path,map_location=device))
        model.eval()
        #------------------------------------------------------------------
        # model = torch.load(model_path, map_location=device)
        # -----------------------------------------------------------------
        plt.rcParams['axes.unicode_minus'] = False
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        dt = 1e-6
        ampfreqs_count = len(ampfreqs)
        fig, axarr = plt.subplots(ampfreqs_count, sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0})
        letter_num = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)',4:'(e)'}
        for i, ampfreq in enumerate(ampfreqs):
            data_path = r'dataset\origin_test_meanv1\\' + str(ampfreq[0]) + '_' + str(ampfreq[1]) + 'Hz.csv'
            data_df = pd.read_csv(data_path)
            data_np = data_df.to_numpy(dtype=np.float32)
            print(data_np.shape, type(data_np), data_np.dtype)

            if ampfreq[1]==100:
                dataclear_x = data_np[-27000:, 0]
                dataclear_y = data_np[-27000:, 1]
            else:
                dataclear_x = data_np[plot_seq_len:, 0]
                dataclear_y = data_np[plot_seq_len:, 1]

            x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
            y = dataclear_y[self.impact_window:]
            # x=torch.from_numpy(x)
            x = x.reshape((x.shape[0], x.shape[1], 1))
            print('x:', x.shape, '\n', 'y:', y.shape, sep='')
            Sampling = sampling(num=self.sample_num, switch_print=self.swit_print, max_length=self.index_maxboundary)
            #variable
            x_index = Sampling.variable_interval(sparsetocompact=self.sparsetocompact)
            x_input_variable = x[:, x_index]
            #equal
            x_index = Sampling.equal_interval()
            x_input_equal = x[:, x_index]
            #
            x_input=np.concatenate((x_input_equal,x_input_variable),axis=2)

            # plot_num = int((2 / ampfreq[1]) / dt)
            plot_num=-1
            x_input = x_input[:plot_num]
            y = y[:plot_num]
            y = y / 0.1792 / ampfreq[0] #0.1792 #0.1078

            with torch.no_grad():
                x_input=torch.tensor(x_input)
                y_pre1 = model(x_input)
                y_pre1 = torch.squeeze(y_pre1) / 0.1792 / ampfreq[0]
                error1 = torch.mean(torch.abs(y_pre1 - y))
                mean = np.mean(np.abs(y))
                re_error1 = error1 / mean
                print("相对误差1dadada：", re_error1)
            x = np.linspace(0, len(y) * dt / (1 / ampfreq[1]), len(y))
            axarr[i].plot(x,y, label='Numerical simulation', linewidth=8, color=(245 / 255, 166 / 255, 115 / 255),
                          alpha=0.5)
            axarr[i].plot(x,y_pre1, label='Neural network', linewidth=4,
                          color=(17 / 255, 50 / 255, 93 / 255),linestyle='--')
            xmin = min(x)
            xmax = max(x)
            ymin = min(min(y), min(y_pre1)) * 1.1
            ymax = max(max(y), max(y_pre1)) * 1.1
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
        fig.text(0.5, 0.03, r'$t/T$', va='center', fontdict=font1,usetex=True)
        fig.text(0.04, 0.5, r"$q^{'}/ \overline{q} /A$", va='center', rotation='vertical', fontdict=font2,usetex=True)
        plt.show()

    def plot_multi_models(self, model_types_paths,ampfreqs,plot_seq_len):
        device = torch.device('cpu')
        # -----------------------------------------------------------------

        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        dt = 1e-6
        ampfreqs_count = len(ampfreqs)
        fig, axarr = plt.subplots(ampfreqs_count, sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0})
        letter_num = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'}
        for i, ampfreq in enumerate(ampfreqs):

            data_path = r'.\dataset\origin_test_meanv1\\' + str(ampfreq[0]) + '_' + str(ampfreq[1]) + 'Hz.csv'
            data_df = pd.read_csv(data_path)
            data_np = data_df.to_numpy(dtype=np.float32)
            print(data_np.shape, type(data_np), data_np.dtype)
            dataclear_x = data_np[plot_seq_len:, 0]
            dataclear_y = data_np[plot_seq_len:, 1]
            x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
            y = dataclear_y[self.impact_window:]
            x = torch.tensor(x)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            print('x:', x.shape, '\n', 'y:', y.shape, sep='')


            Sampling = sampling(num=num, switch_print=self.print_index, max_length=self.impact_window)
            # variable
            x_index = Sampling.variable_interval(sparsetocompact=True)
            x_index=x_index.copy()
            x_short_variable = x[:, x_index]
            # equal
            x_index = Sampling.equal_interval()
            x_short_equal = x[:, x_index]
            #
            x_short = np.concatenate((x_short_equal, x_short_variable), axis=2)

            plot_num = int((2 / ampfreq[1]) / dt)
            x_original = x[:plot_num]
            x_short = x_short[:plot_num]
            y_original = y[:plot_num]
            y_original = y_original / 0.1792 / ampfreq[0]
            y_pres=[]

            for model_type_path in model_types_paths:
                s_inference = time.time()
                model = model_type_path[1]
                model.load_state_dict(torch.load(model_type_path[2], map_location=device))
                model.eval()
                if model_type_path[0]=='MLP':
                    x_input = x_original
                elif model_type_path[0] == 'LSTM':
                    x_input = x_original
                elif model_type_path[0] == 'Single Path':
                    x_input = x_short_equal
                else:
                    x_input = x_short

                x = torch.Tensor(x_input)
                y = torch.Tensor(y_original)
                dataset = data.TensorDataset(x, y)
                data_loader = data.DataLoader(dataset=dataset,
                                              batch_size=256,
                                              shuffle=False,
                                              drop_last=False)
                y_pre_total = []
                y_total = []
                with torch.no_grad():
                    for x, y in tqdm.tqdm(data_loader):
                        y_pre = model(x)
                        y_pre = torch.squeeze(y_pre) / 0.1792 / ampfreq[0]
                        y_total.append(y)
                        y_pre_total.append(y_pre)

                y = torch.cat(y_total)
                y_pre = torch.cat(y_pre_total)
                error = torch.mean(torch.abs(y_pre - y))
                mean = torch.mean(torch.abs(y))
                re_error = error / mean
                print("MRE_" + model_type_path[0], re_error)
                y_pres.append([model_type_path[0], y_pre])
                e_inference = time.time()
                print('Inference time_'+model_type_path[0]+':',e_inference-s_inference)

            x = np.linspace(0, len(y) * dt / (1 / ampfreq[1]), len(y))
            axarr[i].plot(x, y, label='Numerical simulation', linewidth=8, color=(245 / 255, 166 / 255, 115 / 255),alpha=0.5)
            colors=[(115 / 255, 107 / 255, 157 / 255),(17 / 255, 50 / 255, 93 / 255), (166 / 225, 64 / 225, 54 / 255)]
            for model_index,y_pre in enumerate(y_pres):
                axarr[i].plot(x, y_pre[1], label=y_pre[0], linestyle='--',linewidth=4,color=colors[model_index])

            xmin = min(x)
            xmax = max(x)
            ymin = min(min(y), min([min(y_pre[1]) for y_pre in y_pres])) * 1.2
            ymax = max(max(y), max([max(y_pre[1]) for y_pre in y_pres])) * 1.2
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
        fig.legend(lines, labels, loc=[0.2, 0.89], fontsize=10, ncol=4, frameon=False,
                   prop={'family': 'Times New Roman', 'size': 25})

        font1 = {'family': 'Times New Roman', 'size': 30}
        font2 = {'family': 'Times New Roman', 'size': 30}
        fig.text(0.5, 0.03, r'$t/T$', va='center', fontdict=font1,usetex=True)
        fig.text(0.03, 0.5, r"$q^{'}/ \overline{q} /A$", va='center', rotation='vertical', fontdict=font2,usetex=True)

        plt.show()

    def plot_multi_nonlinearity_models(self,model_types,nonlinearities,ampfreq):

        device = torch.device('cpu')
        # -----------------------------------------------------------------

        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        dt = 1e-6
        fig, axarr = plt.subplots(len(nonlinearities), sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0})
        letter_num = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)',4:'(e)'}
        for i,nonlinearity in enumerate(nonlinearities):
            data_path = r'./Strong_nonlinear/Different_datasize_and_nonlinearity/omegac400_tau1_tau3_2e-3/testdata/a1_1_a3_'\
                        +str(nonlinearity)+'_'+str(ampfreq[0])+'_'+str(ampfreq[1])+'.csv'
            data_df = pd.read_csv(data_path)
            data_np = data_df.to_numpy(dtype=np.float32)
            print(data_np.shape, type(data_np), data_np.dtype)
            dataclear_x = data_np[:, 0]
            dataclear_y = data_np[:, 1]
            x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
            y = dataclear_y[self.impact_window:]
            x = torch.tensor(x)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            print('x:', x.shape, '\n', 'y:', y.shape, sep='')

            plot_num = int((2 / ampfreq[1]) / dt)
            x_original = x[:plot_num]
            y_original = y[:plot_num]
            y_original = y_original / 0.1792 / ampfreq[0]
            y_pres = []

            for model_type in model_types:

                if model_type=='MLP':
                    model=model_mlp()
                elif model_type=='LSTM':
                    model=model_lstm()
                else:
                    model=model_dual_path_nonlinear(out_ch=128)

                model_path = r'./Strong_nonlinear/checkpoint(strong_nonlinear)_' + str(nonlinearity) + '/model-' + model_type + '.pth'
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                x_input = x_original
                x = torch.Tensor(x_input)
                y = torch.Tensor(y_original)
                dataset = data.TensorDataset(x, y)
                data_loader = data.DataLoader(dataset=dataset,
                                              batch_size=256,
                                              shuffle=False,
                                              drop_last=False)
                y_pre_total = []
                y_total = []
                with torch.no_grad():
                    for x, y in tqdm.tqdm(data_loader):
                        y_pre = model(x)
                        y_pre = torch.squeeze(y_pre) / 0.1792 / ampfreq[0]
                        y_total.append(y)
                        y_pre_total.append(y_pre)

                y = torch.cat(y_total)
                y_pre = torch.cat(y_pre_total)
                error = torch.mean(torch.abs(y_pre - y))
                mean = torch.mean(torch.abs(y))
                re_error = error / mean
                print("MRE_" + model_type, re_error)
                y_pres.append([model_type, y_pre])

            x = np.linspace(0, len(y) * dt / (1 / ampfreq[1]), len(y))
            axarr[i].plot(x, y, label='Numerical simulation', linewidth=8, color=(245 / 255, 166 / 255, 115 / 255),
                          alpha=0.5)
            colors = [(115 / 255, 107 / 255, 157 / 255), (17 / 255, 50 / 255, 93 / 255), (166 / 225, 64 / 225, 54 / 255)]
            for model_index, y_pre in enumerate(y_pres):
                axarr[i].plot(x, y_pre[1], label=y_pre[0], linestyle='--', linewidth=4, color=colors[model_index])

            xmin = min(x)
            xmax = max(x)
            ymin = min(min(y), min([min(y_pre[1]) for y_pre in y_pres])) * 1.2
            ymax = max(max(y), max([max(y_pre[1]) for y_pre in y_pres])) * 1.2
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
        fig.legend(lines, labels, loc=[0.2, 0.89], fontsize=10, ncol=4, frameon=False,
                   prop={'family': 'Times New Roman', 'size': 25})

        font1 = {'family': 'Times New Roman', 'size': 30}
        font2 = {'family': 'Times New Roman', 'size': 30}
        fig.text(0.5, 0.03, r'$t/T$', va='center', fontdict=font1, usetex=True)
        fig.text(0.03, 0.5, r"$q^{'}/ \overline{q} /A$", va='center', rotation='vertical', fontdict=font2, usetex=True)

        plt.show()

    def plot_MRE_statistics(self,model_type_path,amps,freqs,plot_seq_len):

        def color_map(data, cmap):
            """数值映射为颜色"""

            dmin, dmax = np.nanmin(data), np.nanmax(data)
            cmo = plt.cm.get_cmap(cmap)
            cs, k = list(), 256 / cmo.N

            for i in range(cmo.N):
                c = cmo(i)
                for j in range(int(i * k), int((i + 1) * k)):
                    cs.append(c)
            cs = np.array(cs)
            data = np.uint8(255 * (data - dmin) / (dmax - dmin))

            return cs[data]

        cmap = 'viridis'
        colors = color_map(amps, cmap)
        device = torch.device('cpu')
        # model_backup_path = r'D:\wjw\FDF_pythonproject\checkpoint\model-dualpath_update1_temporal_prior_equ_var-maxsampling-6000-1000-4.0%(best).pth'
        # model_backup = model_dualpath_update1_temporal_prior_equ_var(out_ch=128)
        # model_backup.load_state_dict(torch.load(model_backup_path, map_location=device))
        # model_backup.eval()
        # -----------------------------------------------------------------
        # plt.figure()
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axarr = plt.subplots(1, sharex='col',
                                  gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        ymax_abs=0
        marker_dict = {0.25:'8',0.35:'s',0.45:'p',0.55:'P',0.65:'*',0.75:'h',0.85:'H',0.95:'X'}
        dt=1e-6
        total_error=[]
        s=time.time()
        for i, amp in enumerate(tqdm.tqdm(amps)):
            error_single_amp = []
            for j, freq in enumerate(tqdm.tqdm(freqs)):
                data_path = r'dataset\origin_test_meanv1\\' + str(amp) + '_' + str(freq) + 'Hz.csv'
                data_df = pd.read_csv(data_path)
                data_np = data_df.to_numpy(dtype=np.float32)
                print(data_np.shape, type(data_np), data_np.dtype)

                if freq == 100:
                    dataclear_x = data_np[-27000:, 0]
                    dataclear_y = data_np[-27000:, 1]
                else:
                    dataclear_x = data_np[plot_seq_len:, 0]
                    dataclear_y = data_np[plot_seq_len:, 1]

                x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
                y = dataclear_y[self.impact_window:]
                # x=torch.from_numpy(x)
                x = x.reshape((x.shape[0], x.shape[1], 1))
                print('x:', x.shape, '\n', 'y:', y.shape, sep='')
                Sampling = sampling(num=self.sample_num, switch_print=self.print_index,
                                    max_length=self.impact_window)
                # variable
                x_index = Sampling.variable_interval(sparsetocompact=True)
                x_index = x_index.copy()
                x_short_variable = x[:, x_index]
                # equal
                x_index = Sampling.equal_interval()
                x_short_equal = x[:, x_index]
                #
                x_short = np.concatenate((x_short_equal, x_short_variable), axis=2)

                plot_num = int((1 / freq) / dt)
                x_original = x[:plot_num]
                x_short = x_short[:plot_num]
                y_original = y[:plot_num]
                y_original = y_original / 0.1792 / amp

                # with torch.no_grad():
                #     x_input = torch.tensor(x_input)
                #     # if amp==0.65:
                #     #     y_pre1=model_backup(x_input)
                #     # else:
                #     y_pre1 = model(x_input)
                #     y_pre1 = torch.squeeze(y_pre1) / 0.1792 / amp
                #     error1 = torch.mean(torch.abs(y_pre1 - y))
                #     mean = np.mean(np.abs(y))
                #     re_error1 = error1 / mean
                #     print("MRE：", re_error1)

                model = model_type_path[1]
                model.load_state_dict(torch.load(model_type_path[2], map_location=device))
                model.eval()
                if model_type_path[0] == 'MLP':
                    x_input = x_original
                elif model_type_path[0] == 'LSTM':
                    x_input = x_original
                elif model_type_path[0] == 'Single Path':
                    x_input = x_short_equal
                else:
                    x_input = x_short

                x = torch.Tensor(x_input)
                y = torch.Tensor(y_original)
                dataset = data.TensorDataset(x, y)
                data_loader = data.DataLoader(dataset=dataset,
                                              batch_size=256,
                                              shuffle=False,
                                              drop_last=False)
                y_pre_total = []
                y_total = []
                with torch.no_grad():
                    for x, y in data_loader:
                        y_pre = model(x)
                        y_pre = torch.squeeze(y_pre) / 0.1792 / amp
                        y_total.append(y)
                        y_pre_total.append(y_pre)

                y = torch.cat(y_total)
                y_pre = torch.cat(y_pre_total)
                error = torch.mean(torch.abs(y_pre - y))
                mean = torch.mean(torch.abs(y))
                re_error = error / mean
                print("MRE_" + model_type_path[0], re_error)

                total_error.append(re_error)
                error_single_amp.append(re_error)

            error_single_amp_df = pd.DataFrame(error_single_amp)
            error_single_amp_df.to_csv('error_single_amp_Dual_Path.csv')
            ymax = max(error_single_amp)
            ymax_abs = max(ymax_abs, abs(ymax))
            axarr.scatter(freqs, error_single_amp, marker=marker_dict[amp], s=120,label=str(int(amp*100))+'%',color=colors[i])
            # axarr.legend(fontsize=20,frameon=False,ncol=8,bbox_to_anchor=(0.49, 1.07), loc=9, borderaxespad=0)
            axarr.set_xticks(range(100, 1000, 100))
            axarr.tick_params(labelsize=20, direction='in', length=6, width=2)

            axarr.spines['bottom'].set_linewidth(2)
            axarr.spines['left'].set_linewidth(2)
            axarr.spines['top'].set_linewidth(2)
            axarr.spines['right'].set_linewidth(2)
        e=time.time()
        print('time_inner:',e-s)

        axarr.set_ylim(0, ymax_abs*1.1)
        font1 = {'family': 'Times New Roman', 'size': 30}
        font2 = {'family': 'Times New Roman', 'size': 30}
        # fig.text(0.5, 0.03, '$f$ [Hz]', va='center', fontdict=font1 ,usetex=True)
        # fig.text(0.05, 0.5, "MRE", va='center', rotation='vertical',fontdict=font2)
        print('Average MRE：',np.mean(total_error),'total_error.shape:',len(total_error))
        plt.show()

class Plot_MRE_statistics():

    def __init__(self, amp, frequcy, sampling_type, sparsetocompact, swit_print, swit_plot, impact_window,
                 index_maxboundary, sample_num):
        self.amp = amp
        self.frequcy = frequcy
        self.sampling_type = sampling_type
        self.sparsetocompact = sparsetocompact
        self.swit_print = swit_print
        self.swit_plot = swit_plot
        self.impact_window = impact_window
        self.index_maxboundary = index_maxboundary
        self.sample_num = sample_num

    def plot_MRE_single_model(self, amps, freqs, plot_seq_len, model_path):
        device = torch.device('cpu')
        print('model_path:', model_path)
        model = model_dual_path(out_ch=128)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        # model = torch.load(model_path, map_location=device)
        # -----------------------------------------------------------------
        # plt.figure()
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axarr = plt.subplots(1, sharex='col',
                                  gridspec_kw={'hspace': 0.2, 'wspace': 0.2},figsize=(5,2.5))
        ymax_abs = 0
        dt=1e-6
        s=time.time()
        for i, amp in enumerate(amps):
            error_single_amp = []
            for j, freq in enumerate(freqs):
                data_path = r'dataset\origin_test_meanv0.6\\' + str(amp) + '_' + str(freq) + 'Hz.csv'
                data_df = pd.read_csv(data_path)
                data_np = data_df.to_numpy(dtype=np.float32)
                print(data_np.shape, type(data_np), data_np.dtype)

                if freq == 100:
                    dataclear_x = data_np[-27000:, 0]
                    dataclear_y = data_np[-27000:, 1]
                else:
                    dataclear_x = data_np[plot_seq_len:, 0]
                    dataclear_y = data_np[plot_seq_len:, 1]

                x = np.lib.stride_tricks.sliding_window_view(dataclear_x[1:], self.impact_window, axis=0)
                y = dataclear_y[self.impact_window:]
                # x=torch.from_numpy(x)
                x = x.reshape((x.shape[0], x.shape[1], 1))
                print('x:', x.shape, '\n', 'y:', y.shape, sep='')
                Sampling = sampling(num=self.sample_num, switch_print=self.swit_print,
                                    max_length=self.index_maxboundary)
                # variable
                x_index = Sampling.variable_interval(sparsetocompact=self.sparsetocompact)
                x_input_variable = x[:, x_index]
                # equal
                x_index = Sampling.equal_interval()
                x_input_equal = x[:, x_index]
                #
                x_input = np.concatenate((x_input_equal, x_input_variable), axis=2)

                plot_num = int((2 / freq) / dt)
                x_input = x_input[:plot_num]
                y = y[:plot_num]
                y = y / 0.1078 / amp  # 0.1792 #0.1078

                with torch.no_grad():
                    x_input = torch.tensor(x_input)
                    y_pre1 = model(x_input)
                    y_pre1 = torch.squeeze(y_pre1) / 0.1078 / amp
                    error1 = torch.mean(torch.abs(y_pre1 - y))
                    mean = np.mean(np.abs(y))
                    re_error1 = error1 / mean
                    print("相对误差1dadada：", re_error1)
                error_single_amp.append(re_error1)

            ymax = max(error_single_amp)
            ymax_abs = max(ymax_abs, abs(ymax))
            axarr.scatter(freqs, error_single_amp, marker='*', s=400, label=str(amp),color=(17 / 255, 50 / 255, 93 / 255))
            axarr.set_xticks(range(100, 1000, 100))
            axarr.tick_params(labelsize=20, direction='in', length=6, width=2)

            axarr.spines['bottom'].set_linewidth(2)
            axarr.spines['left'].set_linewidth(2)
            axarr.spines['top'].set_linewidth(2)
            axarr.spines['right'].set_linewidth(2)

        e=time.time()
        print('time:',e-s)

        axarr.set_ylim(0, ymax_abs * 1.1)
        font1 = {'family': 'Times New Roman', 'size': 24}
        font2 = {'family': 'Times New Roman', 'size': 24}
        fig.text(0.5, 0.03, '$f$ [Hz]', va='center',fontdict=font1, usetex=True)
        fig.text(0.05, 0.5, "MRE", va='center', rotation='vertical',fontdict=font2)
        plt.gcf().set_size_inches(15, 7)
        plt.show()

    def plot_MRE_multi_model(self, amps, freqs, plot_seq_len, model_paths):
        device = torch.device('cpu')
        print('model_path:', model_paths)
        # model1 = torch.load(model_path[0], map_location=device)
        # model2 = torch.load(model_path[1],map_location=device)
        model1 = model_dual_path(out_ch=128)
        model1.load_state_dict(torch.load(model_paths[0], map_location=device))
        model1.eval()
        model2 = model_dual_path(out_ch=128)
        model2.load_state_dict(torch.load(model_paths[1], map_location=device))
        model2.eval()
        # model = torch.load(model_path, map_location=device)
        # -----------------------------------------------------------------
        # plt.figure()
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axarr = plt.subplots(1, sharex='col',
                                  gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        ymax_abs = 0

        for i, amp in enumerate(amps):
            error_single1_amp = []
            error_single2_amp = []
            for j, freq in enumerate(freqs):
                data_path = r'dataset\origin_test_meanv1\\' + str(amp) + '_' + str(freq) + 'Hz.csv'
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
                Sampling = sampling(num=num, switch_print=self.swit_print, max_length=self.index_maxboundary)
                # variable
                x_index = Sampling.variable_interval(sparsetocompact=self.sparsetocompact)
                x_input_variable = x[:, x_index]
                # equal
                x_index = Sampling.equal_interval()
                x_input_equal = x[:, x_index]
                #
                x_input = np.concatenate((x_input_equal, x_input_variable), axis=2)

                dt = 1 * 10e-6
                plot_num = int((1 / amp) / dt)
                x_input = x_input[:plot_num]
                y = y[:plot_num]

                # sampling_index = np.linspace(0, len(x_input) - 1, 10, dtype=int)
                # x_input = x_input[sampling_index, :]
                # y = y[sampling_index]

                dt=1e-6
                plot_num = int((2 / freq) / dt)
                x_input = x_input[:plot_num]
                y = y[:plot_num]
                y = y / 0.1792 / amp
                with torch.no_grad():
                    x_input = torch.tensor(x_input)
                    y_pre1 = model1(x_input) / 100
                    y_pre1 = torch.squeeze(y_pre1) / 0.1792 / amp
                    error1 = torch.mean(torch.abs(y_pre1 - y))
                    mean = np.mean(np.abs(y))
                    re_error1 = error1 / mean
                    print("MRE1：", re_error1)

                    y_pre2 = model2(x_input[:plot_num])
                    y_pre2 = torch.squeeze(y_pre2) / 0.1792 / amp
                    error2 = torch.mean(torch.abs(y_pre2 - y))
                    re_error2 = error2 / mean
                    print("MRE2：", re_error2)
                error_single1_amp.append(re_error1)
                error_single2_amp.append(re_error2)

            ymax = max(max(error_single1_amp),max(error_single2_amp))
            ymax_abs = max(ymax_abs, abs(ymax))
            axarr.scatter(freqs, error_single1_amp, marker='*', s=400,color=(17 / 255, 50 / 255, 93 / 255),label='Fine-tuning')
            axarr.scatter(freqs, error_single2_amp, marker='o', s=400,color=(17 / 255, 50 / 255, 93 / 255),label='No fine-tuning')
            axarr.legend(fontsize=16)
            axarr.set_xticks(range(810, 900, 10))
            axarr.tick_params(labelsize=20, direction='in', length=6, width=2)
            axarr.spines['bottom'].set_linewidth(2)
            axarr.spines['left'].set_linewidth(2)
            axarr.spines['top'].set_linewidth(2)
            axarr.spines['right'].set_linewidth(2)

        axarr.set_ylim(0, ymax_abs * 1.1)
        font1 = {'family': 'Times New Roman', 'size': 24}
        font2 = {'family': 'Times New Roman', 'size': 24}
        fig.text(0.5, 0.03, '$f$ [Hz]', va='center', fontdict=font1, usetex=True)
        fig.text(0.03, 0.5, "MRE", va='center', rotation='vertical', fontdict=font2)
        plt.gcf().set_size_inches(15, 7)
        plt.show()


if __name__ == '__main__':
     # test_error=num_sim_testplot(amp=None,frequcy=None,switch_sampling='variable',maxlen=7999.5,reversal=False,swit_print=True,swit_plot=False)
     # test_error.error_statistics(amps_freqs=[[0.1,100],[0.1,200],[0.2,200],[0.3,400],[0.35,300],[0.5,300],
     #                                         [0.5,750],[0.8,200],[0.8,800],[0.9,400]],model_nums=[800,1000,1600,3200,4000,5000],model_type='variable')

     num=1000
     model_path1=r'./Dual_Path_Model\checkpoint\model-dualpath_update1_temporal_prior_equ_var-maxsampling-6000-1000-4.4%(last_epoch).pth'
     model_path2= r'D:\wjw\FDF_pythonproject\checkpoint\model-dualpath_update1_temporal_prior_equ_var-maxsampling-6000-1000-4.0%(best).pth'
     model_path3=r'D:\wjw\FDF_pythonproject\checkpoint(meanv0.6)\model-dualpath_update1_temporal_prior_equ_var-maxsampling-6000-1000-2.0%(best).pth'
     model_path4=r'D:\wjw\FDF_pythonproject\checkpoint(supplement)\model-dualpath_update1_temporal_prior_equ_var-maxsampling(0.1amp_part_parameters)-6000-1000-9.3%(best).pth'

     model_path6 = r'./MLP_LSTM\checkpoint(mlp)\model-mlp--6000--14.2%(last_epoch).pth'
     model_path7 = r'./MLP_LSTM/checkpoint(lstm)/model-lstm--6000--4.5%(best).pth'
     model_path8 = r'./Dual_Path_Model\checkpoint\model-dualpath_update1_temporal_prior_equ_var-maxsampling-6000-1000-4.4%(last_epoch).pth'
     model_type_path1 = ['MLP', model_mlp(), model_path6]
     model_type_path2 = ['LSTM', model_lstm(), model_path7]
     model_type_path3 = ['Dual Path', model_dual_path(out_ch=128), model_path8]

     model_types_paths = [model_type_path1,model_type_path2,model_type_path3]
     testplot=num_sim_testplot(print_index=False,impact_window=6000,sample_num=num)

     s=time.time()
     # testplot.plot_multi_models(model_types_paths=model_types_paths,ampfreqs=[[0.25,200],[0.25,400],[0.25,600],[0.25,800]],plot_seq_len=-16000)
     # testplot.plot_multi_nonlinearity_models(model_types=['MLP','LSTM','Dual-Path'],nonlinearities=[1,2,3,4,5],ampfreq=[0.45,350])
     testplot.plot_MRE_statistics(model_type_path=model_type_path1,amps=[0.25,0.35,0.45,0.55],
                                  freqs=np.arange(100,901,100),plot_seq_len=-16000)
     e=time.time()
     print('time：',e-s)