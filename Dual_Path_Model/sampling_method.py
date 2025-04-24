import numpy as np
import matplotlib as plt

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
                if dn<1:
                    a = a - 1
                else:
                    a = a-dn
            else:
                if dn<1:
                    a = a + 1
                else:
                    a = a+dn
            if a<0:
                break
            else:
                index_list.append(int(a))
            count.append(i)
        index_list = np.array(index_list)
        if sparsetocompact == True:
            index_list = index_list[::-1]
        if self.switch_print == True:
            print('采样索引信息------------------------------------')
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
            print('采样索引信息------------------------------------')
            print('sampling:equal')
            print('index_list:', index_list.shape, sep='')
            print('---------------------------------------------')
        return index_list

if __name__ == '__main__':
    samp=sampling(1000,switch_print=True,max_length=6000)
    print(samp.variable_interval(sparsetocompact=True))
    print(len([num for num in samp.variable_interval(sparsetocompact=True) if num<0]))

