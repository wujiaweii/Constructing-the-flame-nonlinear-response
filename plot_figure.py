import numpy as np
import torch
import pandas as pd
import matplotlib.pylab as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties


def single_amp(data_path):
    df_data=pd.read_csv(data_path)
    data=df_data.to_numpy()
    # return data
    plt.figure(1)
    plt.plot(data[:,0])
    plt.figure(2)
    plt.plot(data[:,1])
    plt.show()

    return data[:,:]

def different_amp_heatreleaserate_comparison(paths):
    data1=single_amp(paths[0])
    data2=single_amp(paths[1])
    data3=single_amp(paths[2])
    dt=1e-6
    t=np.arange(0,dt*len(data1[6000:]),dt)
    fig,ax=plt.subplots()
    ax.plot(t,data1[6000:,1]/0.1792,linewidth=4,color=(183 / 255, 131 / 255, 175 / 255))
    ax.plot(t,data2[6000:,1]/0.1792,linewidth=4,color=(115 / 255, 107 / 255, 157 / 255))
    ax.plot(t,data3[6000:,1]/0.1792,linewidth=4,color=( 54/ 255, 80 / 255, 131 / 255))

    plt.ylabel("$q^{'}/ \overline{q}$", fontsize=35,usetex=True)
    plt.xlabel('$t$ [s]', fontsize=35,usetex=True)
    # plt.axis('off')
    ax.tick_params(labelsize=23, direction='in', length=6, width=2)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.show()

def freq_600Hz(path):
    x_y = pd.read_csv(path, header=0)
    x_y_numpy = np.array(x_y)
    print(x_y_numpy.shape)
    y = x_y_numpy[6000:, 1]/0.1792 #0.1792，0.1078
    x = np.arange(0, len(y) * 1e-6, 1e-6)

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

    ps = np.stack((x, y), axis=1)
    segments = np.stack((ps[:-1], ps[1:]), axis=1)
    norm1 = matplotlib.colors.Normalize(vmin=10, vmax=1000)
    cmap = 'viridis'  # jet, hsv等也是常用的颜色映射方案
    colors = color_map(x[:-1], cmap)

    line_segments = LineCollection(segments, colors=colors, linewidths=4, linestyles='solid', cmap=cmap, norm=norm1)
    font = FontProperties()
    font.set_family('Times New Roman')
    fig, ax = plt.subplots()
    ax.set_xlim(-0.001, 0.051)
    ax.set_ylim(-0.51, 0.51)
    ax.set_yticks(np.arange(-0.5,0.75,0.25))
    ax.add_collection(line_segments)
    cb = fig.colorbar(line_segments, cmap='jet')
    cb.set_ticks([10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    cb.ax.tick_params(labelsize=23)

    # x_major_locator = MultipleLocator(0.02)
    # ax.xaxis.set_major_locator(x_major_locator)
    font = {'family': 'Times New Roman', 'size': 35}

    cb.set_label('$f$ [Hz]', fontdict=font, usetex=True)
    plt.ylabel("$q^{'}/ \overline{q}$", fontsize=35,usetex=True)
    plt.xlabel('$t$ [s]', fontsize=35,usetex=True)
    # plt.axis('off')
    ax.tick_params(labelsize=23, direction='in', length=6, width=2)


    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.gcf().set_size_inches(10, 5)
    plt.show()

def numerical_sim_data_monovisual(path):
    x_y = pd.read_csv(path, header=0)
    x_y_numpy = np.array(x_y)
    print(x_y_numpy.shape)
    x = x_y_numpy[:, 0]
    y = x_y_numpy[:, 1]
    plt.figure(1)
    plt.plot(x,color=(17/255,50/255,93/255),linewidth=2.5)
    plt.tick_params(labelsize=30)
    #plt.axis('off')
    plt.figure(2)
    t=np.arange(0,len(y)*1e-6,1e-6)
    plt.plot(t,y - 0.1792,color=(17/255,50/255,93/255),linewidth=2.5)
    plt.tick_params(labelsize=30)
    plt.ylabel("$q^{'}/$" + '$\overline{q}$', fontsize=30)
    plt.xlabel('$t/s$', fontsize=30)
    #plt.axis('off')
    plt.show()

def numerical_sim_data_visual():
    num_list=range(1,11,1)
    x_total=[]
    y_total=[]
    for i in num_list:
        path=r'.\dataset\origin_train_meanv1\amp'+str(i/10)+'.csv'
        x_y=pd.read_csv(path,header=0)
        x_y_numpy=np.array(x_y)
        print(x_y_numpy.shape)
        x=x_y_numpy[:,0]
        y=x_y_numpy[:,1]
        x_total=np.concatenate((x_total,x))
        y_total=np.concatenate((y_total,y))
    dt=1e-6
    plt.figure(1)
    plt.plot(np.arange(0,len(x_total)*dt,dt),x_total,color=(17/255,50/255,93/255))
    plt.tick_params(labelsize=27)
    plt.ylabel("$u^{'}/$" + '$\overline{u}$', fontsize=30)
    plt.xlabel('$t/s$', fontsize=30)
    plt.figure(2)
    plt.plot(np.arange(0,len(y_total)*dt,dt),y_total)
    plt.tick_params(labelsize=27)
    plt.ylabel("$q^{'}/$" + '$\overline{q}$', fontsize=30)
    plt.xlabel('$t/s$', fontsize=30)
    plt.show()

def numerical_sim_data_visual_practical():
    x_total=[]
    mean_v = 1
    f_start = 10
    f_end = 1000
    t_start = 0
    t_end = 0.05
    dt=1e-6
    t_before=np.arange(t_start-dt*6000,t_start,dt)
    t=np.arange(t_start,t_end,dt)
    k = (f_end - f_start) / (t_end - t_start)
    b = (t_end * f_start - f_end * t_start) / (t_end - t_start)
    amplitude_list=np.arange(0.1,1.1,0.1)
    for amp in amplitude_list :
        # x_before=amp * mean_v * np.cos(2 * np.pi * f_start * t_before)
        x=amp*mean_v*np.cos(2*np.pi*(0.5*k*pow(t,2)+b*t-0.5*k*pow(t_start,2)-b*t_start)+(2*np.pi*f_start*t_start))
        # x=np.concatenate((x_before,x))
        x_total=np.concatenate((x_total,x))
    dt=1e-6
    plt.figure(1)
    plt.plot(np.arange(0,len(x_total)*dt,dt),x_total,color=(17/255,50/255,93/255),linewidth=1)
    plt.tick_params(labelsize=27)
    plt.ylabel("$u^{'}/$" + '$\overline{u}$', fontsize=30)
    plt.xlabel('$t/s$', fontsize=30)
    plt.show()

def pin_die():
    t=np.arange(0,3,0.001)
    b=0
    for i,f_color in enumerate([[1,(183/255,131/255,175/255)],[2,(115/255,107/255,157/255)],[3,(54/255,80/255,131/255)]]):
        a = np.sin(2 * np.pi *f_color[0]* t)
    b=b+a
    plt.figure()
    plt.plot(a,color=f_color[1],linewidth=4)
    plt.axis('off')
    plt.figure(4)
    plt.plot(np.sin(2 * np.pi *4* t),color=(17/255,50/255,93/255),linewidth=4)
    plt.axis('off')
    plt.show()
    b=[]
    for i,f_t_color in enumerate([[1,3,(183/255,131/255,175/255)],[2,6,(115/255,107/255,157/255)],[3,9,(54/255,80/255,131/255)]]):
        t=np.arange(f_t_color[1]-3,f_t_color[1],0.001)
    a = np.sin(2 * np.pi *f_t_color[0]*t )
    b=np.concatenate((b,a))
    plt.figure()
    plt.plot(a,color=f_t_color[2],linewidth=4)
    plt.axis('off')
    plt.figure(4)
    plt.plot(t,a,color=f_t_color[2],linewidth=4)
    plt.axis('off')
    plt.show()

def splicing_data_plot():
    t1 = np.arange(0, 0.1, 0.001)
    t2 = np.arange(0.1, 0.2, 0.0001)
    t3 = np.arange(0.2, 0.3, 0.0001)
    signal1 = 0.5 * np.cos(2 * np.pi * 30 * t1)
    signal2 = 0.5 * np.cos(2 * np.pi * 50 * t2)
    signal3 = 0.5 * np.cos(2 * np.pi * 100 * t3)

    signal_total = [[t1, signal1, 'skyblue'], [t2, signal2, 'cadetblue'], [t3, signal3, 'steelblue']]

    t_cat = np.concatenate((t1, t2, t3))
    signal_cat = np.concatenate((signal1, signal2, signal3))
    for i, sig in enumerate(signal_total):
        plt.figure(i)
        plt.plot(sig[0], sig[1], color=sig[2])
        plt.axis('off')

    plt.figure(3)
    plt.plot(t1, signal1, color='skyblue')
    plt.plot(t2, signal2, color='cadetblue')
    plt.plot(t3, signal3, color='steelblue')
    plt.axis('off')
    plt.show()

def numerical_sim_data_visual_practical():
    x_total=[]
    mean_v = 1
    f_start = 10
    f_end = 1000
    t_start = 0
    t_end = 0.05
    dt=1e-6
    t_before=np.arange(t_start-dt*6000,t_start,dt)
    t=np.arange(t_start,t_end,dt)
    k = (f_end - f_start) / (t_end - t_start)
    b = (t_end * f_start - f_end * t_start) / (t_end - t_start)
    amplitude_list=np.arange(0.1,1.1,0.1)
    for amp in amplitude_list :
        # x_before=amp * mean_v * np.cos(2 * np.pi * f_start * t_before)
        x=amp*mean_v*np.cos(2*np.pi*(0.5*k*pow(t,2)+b*t-0.5*k*pow(t_start,2)-b*t_start)+(2*np.pi*f_start*t_start))
        # x=np.concatenate((x_before,x))
        x_total=np.concatenate((x_total,x))
    dt=1e-6
    fig, axarr = plt.subplots(1, sharex='col',gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    axarr.plot(np.arange(0,len(x_total)*dt,dt),x_total,color=(17/255,50/255,93/255),linewidth=1)

    axarr.tick_params(labelsize=27, direction='in', length=6, width=2)

    axarr.spines['bottom'].set_linewidth(2)
    axarr.spines['left'].set_linewidth(2)
    axarr.spines['top'].set_linewidth(2)
    axarr.spines['right'].set_linewidth(2)

    font1 = {'family': 'Times New Roman', 'size': 30}
    font2 = {'family': 'Times New Roman', 'size': 30}
    fig.text(0.5, 0.04, '$t$ [s]', va='center', fontdict=font1, usetex=True)
    fig.text(0.04, 0.5, "$u^{'}/$" + '$\overline{u}$', va='center', rotation='vertical', fontdict=font2,usetex=True)

    plt.show()

def input_data_generater(mean_v,amp,f1,f2,t1,t2,dt):
    t_stage1=np.arange(0,t1,dt)
    stage1=amp*mean_v*np.cos(2*np.pi*f1*t_stage1)

    k=(f2-f1)/(t2-t1)
    b=(t2*f1-f2*t1)/(t2-t1)
    t_stage2=np.arange(t1,t2,dt)
    stage2=amp*mean_v*np.cos(2*np.pi*(0.5*k*t_stage2**2+b*t_stage2-0.5*k*t1**2-b*t1)+(2*np.pi*f1*t1))

    data_x=np.concatenate((stage1,stage2))[:,None]
    print(data_x.shape)
    data_y=np.zeros(len(data_x))[:,None]
    print(data_y.shape)
    data=np.concatenate((data_x,data_y),axis=1)
    print(data.shape)
    data_df=pd.DataFrame(data)
    data_df.to_csv(r'D:\wjw\FDF_pythonproject\Subspace_RL\dataset\train_target\amp_'+str(amp)+'_0.2s.csv',header=['x','y'],index=False)
    fig, ax = plt.subplots()
    ax.plot(t_stage1, stage1)
    ax.plot(t_stage2, stage2)
    plt.show()



if __name__ == '__main__':
    path1=r'E:\Paper_two\FDF_pythonproject\dataset\origin_train_meanv1\amp0.1.csv'
    path2 = r'E:\Paper_two\FDF_pythonproject\dataset\origin_train_meanv1\amp0.5.csv'
    path3 = r'E:\Paper_two\FDF_pythonproject\dataset\origin_train_meanv1\amp1.0.csv'
    paths=[path1,path2,path3]

    different_amp_heatreleaserate_comparison(paths=paths)
    # path=r'D:\wjw\FDF_pythonproject\Subspace_RL\dataset\train_target\amp_0.6_0.2s.csv'
    # data=single_amp(data_path=path)
    # numerical_sim_data_visual_practical()
    # input_data_generater(mean_v=1, amp=0.65, f1=10, f2=1000, t1=0.006, t2=0.206, dt=1e-6)
