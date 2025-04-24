import copy
import os
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from network import model_mlp,model_lstm

class Train(object):
    def __init__(self,config,train_loader,valid_loader,test_loader):
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.model = None
        self.optimizer = None
        self.Loss = torch.nn.MSELoss()

        # Hyper-parameters
        self.lr = config.lr


        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.model_type = config.model_type
        self.impact_window=config.impact_window

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.plotloss=config.plotloss
        self.build_model()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_model(self,):
        if self.model_type == 'mlp':
            self.model = model_mlp()
        elif self.model_type == 'lstm':
            self.model = model_lstm()

        self.optimizer = optim.AdamW(self.model.parameters(),self.lr)
        self.model.to(self.device)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def train(self):

        # ====================================== Training ===========================================#
        batchs = []
        batch_loss = []
        batchs_val=[]
        val_loss = []
        test_loss = []
        re_errorvaltotal=[]
        re_errortesttotal=[]
        min_error_test=float('inf')
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 80, 95], gamma=0.1)
        a = -100
        print('epoch_num:', self.num_epochs)
        start=time.time()
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch_num, (input, labals) in enumerate(self.train_loader):
                # for name, parameters in self.model.named_parameters():
                #     print(name, ':', parameters.grad)
                self.optimizer.zero_grad()
                input = input.to(self.device)
                labals = labals.to(self.device)
                output = self.model(input)
                loss = self.Loss(output, labals)
                loss.backward()
                self.optimizer.step()
                if batch_num % 100 == 0:
                    a += 100
                    batchs.append(a)
                    batch_loss.append(loss.detach().cpu().numpy())
                    print("epoch{} - batch_size{} - lr{} - loss:{}".format(epoch, batch_num, scheduler.get_last_lr(),
                                                                           loss))
            # ===================================== Validation =====================================================#
            self.model.eval()
            loss_val = 0
            loss_test = 0
            reerror_val = 0
            reerror_test = 0
            with torch.no_grad():
                for batch_num_val, (input, labals) in enumerate(self.valid_loader):
                    input = input.to(self.device)
                    labals=labals.to(self.device)
                    output = self.model(input)
                    loss_val = loss_val+self.Loss(output, labals)
                    error_val = torch.mean(torch.abs(output - labals))
                    mean_val = torch.mean(torch.abs(labals))
                    reerror_val=reerror_val+(error_val / mean_val)

                for batch_num_test, (input, labals) in enumerate(self.test_loader):
                    input = input.to(self.device)
                    labals=labals.to(self.device)
                    output = self.model(input)
                    loss_test = loss_test+self.Loss(output, labals)
                    error_test = torch.mean(torch.abs(output- labals))
                    mean_test = torch.mean(torch.abs(labals))
                    reerror_test=reerror_test+(error_test / mean_test)

            batchs_val.append(a)
            val_loss.append((loss_val/len(self.valid_loader)).detach().cpu().numpy())
            test_loss.append((loss_test/len(self.test_loader)).detach().cpu().numpy())
            re_errorvaltotal.append((reerror_val/len(self.valid_loader)).detach().cpu().numpy())
            re_errortesttotal.append((reerror_test/len(self.test_loader)).detach().cpu().numpy())
            scheduler.step()

            print("----------------")
            print("valitation")
            print("epoch_valitation{} - loss:{} - re_error_val:{}".format(epoch, loss_val/len(self.valid_loader),reerror_val/len(self.valid_loader)))
            print("epoch_mono_test{} - loss:{} - re_error_test:{}".format(epoch, loss_test/len(self.test_loader),reerror_test/len(self.test_loader)))
            print("----------------")
            if reerror_test/len(self.test_loader) < min_error_test:
                min_error_test=reerror_test/len(self.test_loader)
                path = self.model_path + '-' + self.model_type + '-' + '-' + str(
                    self.impact_window) + '-' + \
                       '-' + '%.1f' % (round(float(min_error_test), 3) * 100) + '%(best).pth'
                print('Model saved as %s' % path)
                torch.save(self.model.state_dict(), path)
                print('The present min test error:{}'.format(reerror_test/len(self.test_loader)))

        end=time.time()

        print("验证集损失函数值:",val_loss[-1])
        print("相对误差：", re_errorvaltotal[-1])
        print('\n', '单频测试集损失函数值:', test_loss[-1],sep='')
        print("单频相对误差：", float(min_error_test))
        print("计算时间：",(end-start)/3600)
        path1=self.model_path+'-'+self.model_type+'-'+'-'+str(self.impact_window)+'-'+\
                            '-'+'%.1f'%(round(float(re_errortesttotal[-1]), 3) * 100)+'%(last_epoch).pth'
        torch.save(self.model.state_dict(),path1)
        print("Model saved as %s" % path1)

        if self.plotloss==True:
            fig, ax = plt.subplots()
            ax.plot(batchs, np.log10(batch_loss), label='origin_train_standard')
            ax.plot(batchs_val, np.log10(val_loss), label='val')
            ax.plot(batchs_val, np.log10(test_loss), label='origin_test_standard')
            plt.legend(bbox_to_anchor=(1, 1))
            ax1 = ax.twinx()
            ax1.plot(batchs_val, np.log10(re_errorvaltotal), label='re_errorval', color='darkred')
            ax1.plot(batchs_val, np.log10(re_errortesttotal), label='re_errormonoval', color='darkviolet')
            plt.legend(bbox_to_anchor=(0.8, 1))
            plt.title("Loss")
            plt.show()

