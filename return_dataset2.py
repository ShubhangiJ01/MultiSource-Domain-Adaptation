import os

import torch
from torchvision import transforms

from loaders.data_list import Imagelists_VISDA, return_classlist


class ResizeImage():
    def __init__(self, size):
        print("resizing image")
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def return_dataset(args):
    print("entered return_dataset function")
    base_path = './data/txt'
    image_set_file_s1 = os.path.join(base_path, args.source1 +'_all' + '.txt')
    image_set_file_s2 = os.path.join(base_path, args.source2 +'_all' + '.txt')
    #image_set_file_s2= os.path.join(base_path, args.source2 +'_all' + '.txt')
    image_set_file_t = os.path.join(base_path, args.target + '_all' + '.txt')
    #image_set_file_test = os.path.join(base_path, args.target + '_unl' + '.txt')
    if args.net == 'alexnet':
        print("network is alexnet, crop size is 227")
        crop_size = 227
    else:
        print("network is ",args.net, " crop size is 224")
        crop_size = 224
    print("transforming data")
    data_transforms = {
        'train1': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train2': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("reading datasets")
    source_dataset1= Imagelists_VISDA(image_set_file_s1, transform=data_transforms['train1'])
    source_dataset2 = Imagelists_VISDA(image_set_file_s2, transform=data_transforms['train2'])
    target_dataset = Imagelists_VISDA(image_set_file_t, transform=data_transforms['val'])
    #target_dataset_unl = Imagelists_VISDA(image_set_file_test, transform=data_transforms['val'])

    
    class_list1 = return_classlist(image_set_file_s1)
    class_list2 = return_classlist(image_set_file_s1)
    print("%d classes in this dataset (based on source 1)"%len(class_list1))
    print("%d classes in this dataset (based on source 2)"%len(class_list2))


    if args.net == 'alexnet':
        print("network is alexnet, batch size is 32") 
        bs = 32
    else:
        print("network is ",args.net, " batch size is 24") 
        bs = 24
    print("loading datasets")
    source_loader1 = torch.utils.data.DataLoader(source_dataset1, batch_size=bs, num_workers=3, shuffle=True,
                                                drop_last=True)
    source_loader2 = torch.utils.data.DataLoader(source_dataset2, batch_size=bs, num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)),
                                                num_workers=3, shuffle=True, drop_last=True)
    #target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs * 2, num_workers=3,
                                                    #shuffle=True, drop_last=True)
    print("returning dataset")
    return source_loader1, source_loader2, target_loader, class_list1, class_list2


"""
def return_dataset_test(args):
    base_path = './data/txt'
    image_set_file_s = os.path.join(base_path, args.source +'_all' + '.txt')
    image_set_file_test = os.path.join(base_path, args.target + '_unl' + '.txt')
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, transform=data_transforms['test'],test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset"%len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs * 2, num_workers=3,
                                                    shuffle=False, drop_last=False)
    return target_loader_unl,class_list
"""
