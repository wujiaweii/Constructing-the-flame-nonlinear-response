import time

import numpy as np

from nl_ds_model import *
from data_preparation import nlds_datapre
from data_preparation import nlds_datapre
from data_preparation import nlds_testdata_monofrequcy
from torch import optim, nn
from nl_ds_model import model_dual_path_nonlinear
from torch.utils import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#函数区---------------------------
def loader(x,y):

    shuffle_index_train = np.random.permutation(np.arange(len(x)))
    x = x[shuffle_index_train]
    y = y[shuffle_index_train]
    print('traindata_source_size:', len(x))

    num_train = int(len(x) * 0.95)

    dataset_train = data.TensorDataset(x[:num_train], y[:num_train])

    dataset_val = data.TensorDataset(x[num_train:], y[num_train:])

    Dataloaders_train = data.DataLoader(dataset=dataset_train,
                                        batch_size=256,
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True)

    Dataloaders_val = data.DataLoader(dataset=dataset_val,
                                      batch_size=256,
                                      shuffle=True,
                                      num_workers=4,
                                      drop_last=True)

    return Dataloaders_train,Dataloaders_val

def train_model(a1,a3,num,amp,freq,seqlen,epoch,l_r):
    def train(Epoch, lr):
        optimizer = optim.AdamW(model.parameters(), lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 90], gamma=0.1)
        a = -20
        for epoch in range(Epoch):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            for batch_num, (input, labals) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                input = input.cuda()
                labals = labals.cuda()
                output = model(input)
                loss = Loss(output, labals)
                loss.backward()
                optimizer.step()
                if batch_num % 20 == 0:
                    a += 20
                    print("epoch{} - batch_size{} - lr{} - loss:{}".format(epoch, batch_num, scheduler.get_last_lr(),
                                                                           loss))

            model.eval()
            with torch.no_grad():
                loss_val = 0
                reerror_val = 0
                count_val = len(val_loader)
                for batch_num_val, (input, labals) in enumerate(val_loader):
                    input = input.to(device)
                    labals = labals.to(device)
                    output = model(input)
                    loss_val = loss_val + Loss(output, labals)
                    error = torch.mean(torch.abs(output - labals))
                    mean = torch.mean(torch.abs(labals))
                    reerror_val = reerror_val + (error / mean)
                loss_val = (loss_val / count_val).detach().cpu().numpy()
                reerror_val = (reerror_val / count_val).detach().cpu().numpy()

                loss_test = 0
                reerror_test = 0
                count_test = len(test_loader)
                for batch_num_test, (input, labals) in enumerate(test_loader):
                    input = input.to(device)
                    labals = labals.to(device)
                    output = model(input)
                    loss_test = loss_test + Loss(output, labals)
                    error = torch.mean(torch.abs(output - labals))
                    mean = torch.mean(torch.abs(labals))
                    reerror_test = reerror_test + (error / mean)
                loss_test = (loss_test / count_test).detach().cpu().numpy()
                reerror_test = (reerror_test / count_test).detach().cpu().numpy()

            print("----------------------------------------------------------------------")
            print("valitation")
            print("epoch_valitation{} - loss:{} - re_error_val:{}".format(epoch, loss_val, reerror_val))
            print("epoch_test{} - loss:{} - re_error_val:{}".format(epoch, loss_test, reerror_test))
            print("------------------------------------------------")
            scheduler.step()

        path = 'checkpoint(strong_nonlinear)_Dual_Path/model_a1_' + str(a1) + '_a3_' + str(a3) + '_' + str(
            num) + '_' + str(round(float(reerror_test), 3) * 100) + '.pth'
        # torch.save(model, path)
        torch.save(model.state_dict(), path)

        print("Model saved as %s" % path, '\n')

    path_train=r'./Different_datasize_and_nonlinearity\omegac400_tau1_tau3_2e-3\traindata\sweep_a1_'+str(a1)+'_a3_'+str(a3)+'_'+str(num)+'.csv'
    path_mono = r'./Different_datasize_and_nonlinearity\omegac400_tau1_tau3_2e-3\testdata\a1_'+str(a1)+'_a3_'+str(a3)+'_'+str(amp)+'_'+str(freq)+'.csv'
    x_load,y_load=nlds_datapre(path=path_train,seq_len=seqlen)
    train_loader,val_loader= loader(x_load, y_load)
    test_loader=nlds_testdata_monofrequcy(seqlen,a1,a3)
    model = model_dual_path_nonlinear(out_ch=128)
    model = model.cuda()
    Loss = nn.MSELoss()
    train(Epoch=epoch, lr=l_r) #Epoch=100,lr=0.0001
    #---------------------------------


#函数区---------------------------
if __name__ =='__main__':
    a1 = 1
    amp = 0.45
    freq = 100
    seqlen = 225
    # train_model(a1=a1,a3=a3,num=num,amp=amp,freq=freq,seqlen=seqlen,epoch=100,l_r=0.0001)
    # -----------------------------
    a3s=range(1,6,1)
    nums=range(20000,100001,20000)

    for a3 in a3s:
        for num in nums:
            train_model(a1=a1,a3=a3,num=num,amp=amp,freq=freq,seqlen=seqlen,epoch=100,l_r=0.0001)