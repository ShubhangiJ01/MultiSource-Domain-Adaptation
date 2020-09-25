import numpy as np
from scipy.io import loadmat

base_dir = './data'
base_path = './data/txt'

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


def load_real(scale=True, usps=False, all_use=False):
    #real_data = loadmat(base_dir + '/real/.mat')
    print("entered load_dataset function for real images")
    image_set_file_s = os.path.join(base_path, args.source +'_all' + '.txt')
    image_set_file_t = os.path.join(base_path, args.target + '_labeled' + '.txt')

    if args.net == 'alexnet':
        print("network is alexnet, crop size is 227")
        crop_size = 227
    else:
        print("network is ",args.net, " crop size is 224")
        crop_size = 224
    print("transforming data")
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
    print("reading datasets")
    source_dataset = Imagelists_VISDA(image_set_file_s, transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, transform=data_transforms['val'])

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset"%len(class_list))

    if args.net == 'alexnet':
        print("network is alexnet, batch size is 32") 
        bs = 32
    else:
        print("network is ",args.net, " batch size is 24") 
        bs = 24
    print("loading datasets")
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs, num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)),
                                                num_workers=3, shuffle=True, drop_last=True)
    print("returning dataset")

    return source_loader, target_loader, target_loader_unl,class_list
    """code to load from data/real/"""



    """
    if scale
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test =  mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    
    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:25000]
    test_label = test_label[:25000]
    print('mnist train X shape->',  mnist_train.shape)
    print('mnist train y shape->',  train_label.shape)
    print('mnist test X shape->',  mnist_test.shape)
    print('mnist test y shape->', test_label.shape)


    """

    return real_train, train_label, real_test, test_label
