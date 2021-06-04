import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from datasets.mscoco2k_segment import MSCOCO2kSegmentDataset, MSCOCO2kSegmentPreprocessor
from datasets.mscoco2k_segment_image import MSCOCO2kSegmentImageDataset, MSCOCO2kSegmentImagePreprocessor
from datasets.speechcoco_segment import SpeechCOCOSegmentDataset, SpeechCOCOSegmentPreprocessor
from datasets.flickr8k import FlickrSegmentDataset, FlickrSegmentPreprocessor
from datasets.flickr8k_word_image import FlickrWordImageDataset, FlickrWordImagePreprocessor

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
    elif 'MSCOCO2K' == name :
        preprocessor = MSCOCO2kSegmentPreprocessor(dset_dir, 80, level='word')
        train_data = MSCOCO2kSegmentDataset(dset_dir,
                             preprocessor, 'train')
        test_data = MSCOCO2kSegmentDataset(dset_dir,
                            preprocessor, 'test')
    elif 'MSCOCO2K_SEGMENT_IMAGE' in name :
        preprocessor = MSCOCO2kSegmentImagePreprocessor(dset_dir, 80, level='word')
        train_data = MSCOCO2kSegmentImageDataset(dset_dir,
                             preprocessor, 'train')
        test_data = MSCOCO2kSegmentImageDataset(dset_dir,
                            preprocessor, 'test')
    elif 'SPEECHCOCO' in name:
        preprocessor = SpeechCOCOSegmentPreprocessor(dset_dir, 80)
        train_data = SpeechCOCOSegmentDataset(dset_dir,
                                              preprocessor, 'train')
        test_data = SpeechCOCOSegmentDataset(dset_dir,
                                              preprocessor, 'test')
    elif 'FLICKR' == name:
        preprocessor = FlickrSegmentPreprocessor(dset_dir, 80)
        train_data = FlickrSegmentDataset(dset_dir,
                                          preprocessor, 'train')
        test_data = FlickrSegmentDataset(dset_dir,
                                         preprocessor, 'test') 
    elif 'FLICKR_WORD_IMAGE' == name :
        preprocessor = FlickrWordImagePreprocessor(dset_dir, 80,
                                                   audio_feature=args.audio_feature, 
                                                   image_feature=args.image_feature,
                                                   min_class_size=args.min_class_size)
        train_data = FlickrWordImageDataset(dset_dir,
                                            preprocessor, 
                                            'train',
                                            audio_feature=args.audio_feature,  
                                            image_feature=args.image_feature,
                                            min_class_size=args.min_class_size,
                                            use_segment=args.use_segment,
                                            ds_method=args.downsample_method)
        test_data = FlickrWordImageDataset(dset_dir,
                                           preprocessor, 
                                           'test',
                                           audio_feature=args.audio_feature,
                                           image_feature=args.image_feature,
                                           min_class_size=args.min_class_size,
                                           use_segment=args.use_segment,
                                           ds_method=args.downsample_method) 
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
