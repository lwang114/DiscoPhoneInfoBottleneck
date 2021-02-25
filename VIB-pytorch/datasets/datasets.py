import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from datasets.mscoco2k_segment import Dataset, Preprocessor

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size

    if 'MNIST' in name :
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),])
        root = os.path.join(dset_dir,'MNIST')
        train_kwargs = {'root':root,'train':True,'transform':transform,'download':True}
        test_kwargs = {'root':root,'train':False,'transform':transform,'download':False}
        dset = MNIST
        train_data = dset(**train_kwargs)
        test_data = dset(**test_kwargs)    
    elif 'MSCOCO2K' in name :
        preprocessor = Preprocessor(dset_dir, 80, level='word')
        train_data = Dataset(dset_dir,
                             preprocessor, 'train')
        test_data = Dataset(dset_dir,
                            preprocessor, 'test')
    else : raise UnknownDatasetError()


    train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)

    test_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)

    data_loader = dict()
    data_loader['train']=train_loader
    data_loader['test']=test_loader

    return data_loader


if __name__ == '__main__' :
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--dset_dir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    data_loader = return_data(args)
    import ipdb; ipdb.set_trace()
