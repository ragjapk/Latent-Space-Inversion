import os
import torch
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from data.default import pacs_path
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset, ConcatDataset, random_split
import bisect
transform_train = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

pacs_name_dict = {
    'p': 'photo',
    'a': 'art_painting',
    'c': 'cartoon',
    's': 'sketch',
}

split_dict = {
    'train': 'train',
    'val': 'crossval',
    'total': 'test',
}


class PACS_SingleDomain():
    def __init__(self, root_path=pacs_path, domain_name='p', db=0, split='total', train_transform=None):
        if domain_name in pacs_name_dict.keys():
            self.domain_name = pacs_name_dict[domain_name]
            self.domain_label = db
            
        else:
            raise ValueError('domain_name should be in p a c s')
        
        self.root_path = os.path.join(root_path, 'PACS')
        self.split = split
        self.split_file = os.path.join(root_path, 'PACS', f'{self.domain_name}_{split_dict[self.split]}_kfold' + '.txt')
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = transform_test
                
        imgs, labels = PACS_SingleDomain.read_txt(self.split_file, self.root_path)
        self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)
        
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
    
class PACS_FedDG():
    def __init__(self, test_domain='p', batch_size=16):
        self.batch_size = batch_size
        self.domain_list = list(pacs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}

        for idx, domain in enumerate(sorted(self.train_domain_list)):
            self.site_dataloader_dict[domain], self.site_dataset_dict[domain] = PACS_FedDG.SingleSite(domain, idx, self.batch_size)

        self.site_dataloader_dict[test_domain], self.site_dataset_dict[test_domain] = PACS_FedDG.SingleSite(test_domain, idx+1, self.batch_size)

        '''for domain_name in self.domain_list:
            if (domain_name == 'p' and test_domain =='a') or (domain_name=='a' and test_domain=='p') or (domain_name=='p' and test_domain=='s') or (domain_name=='p' and test_domain=='c'): 
                self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = PACS_FedDG.SingleSite(domain_name, 0, self.batch_size)
            elif (domain_name == 'c' and test_domain =='a') or (domain_name=='c' and test_domain=='p') or (domain_name=='a' and test_domain=='s') or (domain_name=='a' and test_domain=='c'): 
                self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = PACS_FedDG.SingleSite(domain_name, 1, self.batch_size)
            elif (domain_name == 's' and test_domain =='a') or (domain_name=='s' and test_domain=='p') or (domain_name=='c' and test_domain=='s') or (domain_name=='s' and test_domain=='c'): 
                self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = PACS_FedDG.SingleSite(domain_name, 2, self.batch_size)
            elif domain_name == test_domain: 
                self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = PACS_FedDG.SingleSite(domain_name, 3, self.batch_size)
        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']'''
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
          
    @staticmethod
    def SingleSite(domain_name, db, batch_size=16):
        dataset_dict = {
            'train': PACS_SingleDomain(domain_name=domain_name, db=db, split='train', train_transform=transform_train).dataset,
            'val': PACS_SingleDomain(domain_name=domain_name, db=db, split='val').dataset,
            'test': PACS_SingleDomain(domain_name=domain_name, db=db, split='total').dataset,
        }
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict
    
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
    
class PACS_DG():
    def __init__(self, test_domain='p', batch_size=16):
        self.batch_size = batch_size
        self.domain_list = list(pacs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  

        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for idx, domain in enumerate(sorted(self.train_domain_list)):
            self.site_dataset_dict[domain] = PACS_DG.SingleSite(domain, idx, self.batch_size)
        
        self.site_dataset_dict[test_domain] = PACS_DG.SingleSite(test_domain, idx+1, self.batch_size)

        '''for domain_name in self.domain_list:
            if (domain_name == 'p' and test_domain =='a') or (domain_name=='a' and test_domain=='p') or (domain_name=='p' and test_domain=='s') or (domain_name=='p' and test_domain=='c'): 
                self.site_dataset_dict[domain_name] = PACS_DG.SingleSite(domain_name, 0, self.batch_size)
            elif (domain_name == 'c' and test_domain =='a') or (domain_name=='c' and test_domain=='p') or (domain_name=='a' and test_domain=='s') or (domain_name=='a' and test_domain=='c'): 
                self.site_dataset_dict[domain_name] = PACS_DG.SingleSite(domain_name, 1, self.batch_size)
            elif (domain_name == 's' and test_domain =='a') or (domain_name=='s' and test_domain=='p') or (domain_name=='c' and test_domain=='s') or (domain_name=='s' and test_domain=='c'): 
                self.site_dataset_dict[domain_name] = PACS_DG.SingleSite(domain_name, 2, self.batch_size)
            elif domain_name == test_domain: 
                self.site_dataset_dict[domain_name] = PACS_DG.SingleSite(domain_name, 3, self.batch_size)'''
        
        #self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
        self.dataset_dict = {'train':ConcatDataset([self.site_dataset_dict[domain_name]['train'] for domain_name in self.train_domain_list]), 
                        'val':ConcatDataset([self.site_dataset_dict[domain_name]['val'] for domain_name in self.train_domain_list]),
                        'test':self.site_dataset_dict[self.test_domain]['test']}
        self.site_dataloader_dict = PACS_DG.SingleSite2(self.dataset_dict, batch_size)
          
    @staticmethod
    def SingleSite(domain_name, db, batch_size=16):
        dataset_dict = {
            'train': PACS_SingleDomain(domain_name=domain_name, db=db, split='train', train_transform=transform_train).dataset,
            'val': PACS_SingleDomain(domain_name=domain_name, db = db, split='val').dataset,
            'test': PACS_SingleDomain(domain_name=domain_name, db=db, split='total').dataset,
        }
        return dataset_dict
    
    @staticmethod
    def SingleSite2(dataset_dict, batch_size=16):
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict, self.dataset_dict
    







