import os 
import torch
import torchvision

import torchvision.models as models
from torchvision import transforms
from torch import optim, nn 
import PIL 

from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import utils 

ROOT = "/data/t-hashah/pytorch_datasets/imagenet/"

def get_imagenet_model(model, pretrained=False, num_classes=2, num_channels=3, **kw):

    nc = num_classes
    
    mods = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'mobilenet': models.mobilenet_v2,
        'shufflenet': models.shufflenet_v2_x1_0,
        'densenet': models.densenet121,
        'alexnet': models.alexnet,
        'resnext50': models.resnext50_32x4d
    }

    assert model in mods.keys(), 'no such model: {}'.format(model)
    margs = dict(pretrained=True)
    if model == 'densenet': margs['memory_efficient'] = True
    m = mods[model](**margs)

    if num_channels != 3:
        if 'resnet' in model:
            m.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif model == 'shufflenet':
            m.conv1[0] = nn.Conv2d(num_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        elif model == 'mobilenet':
            m.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        elif model == 'densenet':
            m.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif model == 'alexnet':
            m.features[0] =  nn.Conv2d(num_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        elif model == 'resnext50':
            m.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if pretrained:
        for p in m.parameters():
            p.requires_grad = False

    # change linear layer
    if model in ['resnet18', 'resnet34']:
        m.fc = nn.Linear(512, nc)
        m.fc.requires_grad = True
    if model == 'resnet50':
        m.fc = nn.Linear(2048, nc)
        m.fc.requires_grad = True
    if model == 'shufflenet':
        m.fc = nn.Linear(1024, nc)
        m.fc.requires_grad = True
    if model == 'mobilenet':
        m.classifier[1] = nn.Linear(1280, nc)
        for p in m.classifier.parameters():
            p.requires_grad = True
    if model == 'densenet':
        m.classifier = nn.Linear(in_features=1024, out_features=nc, bias=True)
    if model == 'alexnet':
        m.classifier[6] = nn.Linear(in_features=4096, out_features=nc, bias=True)
    if model == 'resnext50':
        m.fc = nn.Linear(in_features=2048, out_features=nc, bias=True)

    return m

def get_imagenet_labels():
    invert_map = lambda d: {v:k for k,v in d.items()}
    folder2id = torch.load(os.path.join(ROOT, 'folder2id.pkl'))
    folder2name = torch.load(os.path.join(ROOT, 'folder2name.pkl'))
    id2name = {v: folder2name[k] for k, v in folder2id.items()}

    return {
        'folder2id': folder2id,
        'folder2name': folder2name,
        'id2folder': invert_map(folder2id),
        'name2folder': invert_map(folder2name),
        'id2name': id2name,
        'name2id': invert_map(id2name)
    }

def get_custom_imagenet_loaders(dataset_name, augment_type='standard', image_size=112, num_workers=4, batch_size=256, 
                                center_crop_ratio=None, random_crop_scale=(0.08, 1.00), root=ROOT):
    """
    https://robustness.readthedocs.io/en/latest/example_usage/custom_imagenet.html
    {living_9, mixed_10, mixed_13, geirhos_16, big_12}
    """
    # load imagenet hierarchy
    in_path = os.path.join(root, 'ILSVRC/Data/CLS-LOC')
    in_info_path = root
    in_hier = ImageNetHierarchy(in_path, in_info_path)

    # obtain grouping + grouped dataset
    assert dataset_name in {'living_9', 'mixed_10', 'mixed_13', 'geirhos_16', 'big_12'}
    superclass_wnid = common_superclass_wnid(dataset_name)
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
    custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
    mean, std = custom_dataset.mean, custom_dataset.std

    # basic vs. none train data aug
    if augment_type == 'basic':
        tr_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(image_size, image_size), interpolation=PIL.Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()  
        ])

        te_transforms = transforms.Compose([
            transforms.Resize(size=image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.ToTensor()
        ])

    elif augment_type == 'standard':
        center_crop_ratio = 256./224. if center_crop_ratio is None else center_crop_ratio
        super_img_size = int(round(center_crop_ratio*image_size))

        tr_transforms = transforms.Compose([
            transforms.Resize(super_img_size),
            transforms.RandomResizedCrop(size=(image_size, image_size), scale=random_crop_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor()  
        ])

        te_transforms = transforms.Compose([
            transforms.Resize(super_img_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    elif augment_type == 'none':
        center_crop_ratio = 256./224. if center_crop_ratio is None else center_crop_ratio
        super_img_size = int(round(center_crop_ratio*image_size))

        tr_transforms = transforms.Compose([
            transforms.Resize(size=super_img_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.ToTensor()
        ])

        te_transforms = transforms.Compose([
            transforms.Resize(size=super_img_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.ToTensor()
        ])

    else:
        assert False, "invalid augmentation type"

    custom_dataset.transform_train = tr_transforms
    custom_dataset.transform_test = te_transforms

    print ("Augmentation type: {}".format(augment_type))
    print (custom_dataset.transform_train)
    print (custom_dataset.transform_test)
    
    # make loaders
    train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, batch_size=batch_size, 
                                                            val_batch_size=batch_size, data_aug=True)

    metadata = {
        'class_ranges': class_ranges, 
        'label_map': label_map,
        'dataset': custom_dataset
    }

    return train_loader, test_loader, metadata