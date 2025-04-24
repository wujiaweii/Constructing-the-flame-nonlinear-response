import argparse
from dataset_loader import get_loader
from train import Train
from dataset import Dataset
def main(config):
    print(config)
    x_train,y_train,x_valid,y_valid,x_test,y_test=Dataset(config)

    train_loader = get_loader(inputdata=x_train,
                              label=y_train,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers)

    valid_loader = get_loader(inputdata=x_valid,
                              label=y_valid,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers)

    test_loader = get_loader(inputdata=x_test,
                              label=y_test,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers)

    solver = Train(config, train_loader, valid_loader, test_loader)
    solver.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_path',type=str,default='checkpoint(inlet_velocity_1)/model')
    parser.add_argument('--model_type', type=str, default='dual_path',help='dual_path/single_path')

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)  #0.0001
    parser.add_argument('--out_channels',type=int,default=128)
    parser.add_argument('--plotloss',type=bool,default=False)
    parser.add_argument('--fine_tuning_or_not',type=bool,default=False)

    # dataset
    parser.add_argument('--origin_train_data', type=str, default='./dataset/train_inlet_velocity_1/')
    parser.add_argument('--origin_test_data', type=str, default='./dataset/test_inlet_velocity_1/')
    parser.add_argument('--sampling_number', type=int, default=100)
    parser.add_argument('--sparsetocompact', type=bool, default=True)
    parser.add_argument('--print_index', type=bool, default=True)
    parser.add_argument('--impact_window', type=int, default=6000)
    parser.add_argument('--singlefreq_test_number', type=int, default=1200)
    parser.add_argument('--train_ratio',type=float,default=0.95)

    # %%
    config = parser.parse_args()
    # %%
    main(config)