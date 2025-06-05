import os
import torch
from data.meta_dataset import GetDataLoaderDict
from data.default import pacs_path
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset, ConcatDataset, random_split, Dataset
import bisect
transform_train = transforms.Compose(
            [#transforms.Resize([224, 224]),
            #transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomGrayscale( 0.1),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            #transforms.RandomGrayscale(),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [#transforms.Resize([224, 224]),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

domain_name_dict = {
    'c': 'clipart',
    'i' : 'infograph',
    'p': 'painting',
    'q' : 'quickdraw',
    'r': 'real',
    's': 'sketch'
}

office_name_dict = {
    'a': 'art',
    'c': 'clipart',
    'p': 'photo',
    'r': 'real-world',
}

pacs_name_dict = {
    'a': 'art_painting',
    'c': 'cartoon',
    'p': 'photo',
    's': 'sketch',
}

vlcs_name_dict = {
    'c': 'caltech101',
    'l': 'labelme',
    's': 'sun09',
    'v': 'voc2007',
}

split_dict = {
    'train': 'train',
    'val': 'crossval',
    'total': 'test',
}

class DomainDataset(Dataset):
    '''
    For RGB data, single client
    '''
    def __init__(self, imgs, labels, domain_label):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
    
    def __getitem__(self, index):
        img = self.imgs[index]
        img_class_label = self.labels[index]
        img_domain_label = self.domain_label[index]
        return img, img_class_label, img_domain_label

    def __len__(self):
        return len(self.imgs)

class PACS_SingleDomain():
    def __init__(self, dataset, test_domain, root_path='latents/latent3', split='total', versions=5, version_index=5):
        if dataset=='PACS':
            domain_list = pacs_name_dict
        elif dataset=='OfficeHome':
            domain_list = office_name_dict
        elif dataset == 'DomainNet':
            domain_list = domain_name_dict
        else:
            domain_list = vlcs_name_dict
        
        final_images, final_labels, final_domains = [], [], []
        if split=='total' or split=='test':
            for i in range(0,versions):
                path = f'{root_path}_{dataset}_{test_domain}_{i}.pt'
                images, labels, domains = torch.load(path)
                final_images.extend(images)
                final_labels.extend(labels)
                final_domains.extend(domains)
        elif split=='val':
            for i in range(0,versions):
                if i in version_index:
                    path = f'{root_path}_{dataset}_{test_domain}_{i}.pt'
                    images, labels, domains = torch.load(path)
                    final_images.extend(images)
                    final_labels.extend(labels)
                    final_domains.extend(domains)
        elif split=='train':
            for i in range(0,versions):
                if i not in version_index:
                    path = f'{root_path}_{dataset}_{test_domain}_{i}.pt'
                    images, labels, domains = torch.load(path)
                    final_images.extend(images)
                    final_labels.extend(labels)
                    final_domains.extend(domains)
        self.dataset = DomainDataset(final_images, final_labels, final_domains)
        
    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            labels.append(int(line_txt[1]) - 1)
        return imgs, labels
    
class PACS_DG():
    def __init__(self, dataset, paths='latent\l', test_domain='p', batch_size=16, versions=5, version_index=5):
        self.batch_size = batch_size
        self.paths = paths
        if dataset=='PACS':
            self.domain_list = list(pacs_name_dict.keys())
        elif dataset == 'OfficeHome':
            self.domain_list = list(office_name_dict.keys())
        elif dataset == 'DomainNet':
            self.domain_list = list(domain_name_dict.keys())
        else:
            self.domain_list = list(vlcs_name_dict.keys())
        self.test_domain = test_domain
        self.dataset=dataset
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain) 

        self.versions = versions
        self.version_index = version_index
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            if domain_name in self.train_domain_list:
                _, self.site_dataset_dict[domain_name] = PACS_DG.SingleSite(self.dataset, self.paths, domain_name, self.test_domain, self.batch_size, self.versions, self.version_index)
            #else:
            #_, self.site_dataset_dict[domain_name] = PACS_DG.SingleSite(domain_name, 3, self.batch_size, self.versions, self.version_index)

        self.dataset_dict = {'train':ConcatDataset([self.site_dataset_dict[domain_name]['train'] for domain_name in self.train_domain_list]), 
                        'val':ConcatDataset([self.site_dataset_dict[domain_name]['val'] for domain_name in self.train_domain_list]),
                        'test':ConcatDataset([self.site_dataset_dict[domain_name]['val'] for domain_name in self.train_domain_list])}

        self.dataloader_dict = PACS_DG.SingleSite2(self.dataset_dict, batch_size)
        
    @staticmethod
    def SingleSite(dataset, path, domain_name, test_domain, batch_size=16, versions=5, version_index=5):
        dataset_dict = {
            'train': PACS_SingleDomain(dataset, test_domain, root_path=path, split='train', versions=versions, version_index=version_index).dataset,
            'val': PACS_SingleDomain(dataset,  test_domain,root_path=path, split='val', versions=versions, version_index=version_index).dataset,
            'test':PACS_SingleDomain(dataset,  test_domain, root_path=path, split='test', versions=versions, version_index=version_index).dataset
        }
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict
    
    @staticmethod
    def SingleSite2(dataset_dict, batch_size=16):
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict
        
    def GetData(self):
        return self.dataloader_dict, self.dataset_dict
    
class ConcatDomainDataset(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        datapoint = self.datasets[dataset_idx][sample_idx]
        return datapoint + (dataset_idx,)
   