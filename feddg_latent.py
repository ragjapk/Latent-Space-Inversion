import argparse
import torch
import random
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import os
import copy
import torch.nn.functional as F
import gc
from torch.utils.data import TensorDataset, Subset, ConcatDataset, random_split, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from utils.log_utils import *
from data.default import pacs_domain_list as domain_list
import data.pacs_dataset as pacsdataset
import data.fake_dataset_latent as fakedataset
import torchvision.models as models
from classification_metric import Classification
from utils.utils import load_model_pytorch, distributed_is_initialized, SaveCheckPoint
from deepinversion import DeepInversionClass
import torch.cuda.amp as amp
from solver_stargan_latent import Solver
import torch.nn as nn
from models.starganl import *
from fed_merge import FedAvg, FedAvg2, Cal_Weight_Dict, FedUpdate
#Should it be the best validation accuracy?

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0])
        var = input[0].permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False)
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def mscatter(x,y, ax=None, m=None, c=None,label=None, alpha=None):
        import matplotlib.markers as mmarkers
        fig, ax = plt.subplots()
        for i in range(len(x)):
            sc = ax.scatter(x[i],y[i],color=c[i],alpha=alpha)
            if (m[i] is not None):
                paths = []
                for marker in m[i]:
                    if isinstance(marker, mmarkers.MarkerStyle):
                        marker_obj = marker
                    else:
                        marker_obj = mmarkers.MarkerStyle(marker)
                    path = marker_obj.get_path().transformed(
                                marker_obj.get_transform())
                    paths.append(path)
                sc.set_paths(paths)
        return sc, ax

class f(nn.Module):
    def __init__(self, encoder, classifier):
        super(f, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
    def forward(self,x):
        self.z = self.encoder(x)
        self.logits = self.classifier(self.z) 
        return self.z, self.logits

def calc_imp_weights(model, dataloader, optimizer):
    model.eval()
    omega={}
    total_points = 0
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()
        optimizer.zero_grad()
        z,outputs = model(imgs) 
        Target_zeros=torch.zeros(outputs.size()).cuda()
        Target_zeros=Target_zeros.cuda()
        #note no avereging is happening here
        loss = torch.nn.MSELoss(reduction='sum')
        targets = loss(outputs,Target_zeros) 
        #compute the gradients
        targets.backward()
        for name, params in model.named_parameters():
            if name not in omega:
                omega[name] = torch.abs(params.grad.data.detach())[:]
            else:
                omega[name] += torch.abs(params.grad.data.detach())[:]
        total_points += len(imgs)
    averaged_omega = {name: grad / (total_points) for name, grad in omega.items()} 
    return averaged_omega

def site_calc_imp(model, optimizer, dataloader):
    omega = {}
    omega = calc_imp_weights(model, dataloader, optimizer)
    return omega

def train_local_validate(epochs, model, dataloader, metric):
    model.eval()
    metric = Classification()
    with torch.no_grad():
        for imgs, labels, domains in dataloader:
            imgs = imgs.cuda()
            z,output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict

def train_local(round, args, domain,  model, optimzier, train_dataloader, val_dataloader, criterion, metric):
    best_val_acc = 0.0
    best_model = None
    best_epoch = 0
    for epoch in range(args.local_epochs):
        model.train()
        metric = Classification()
        for i, data_list in enumerate(train_dataloader):
            imgs, labels, domain_labels = data_list
            imgs = imgs.cuda()
            labels = labels.cuda()
            optimzier.zero_grad()
            z,output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimzier.step()
            metric.update(output, labels)
        results_dict = metric.results()
        args.log_file.info(f'Epoch: {epoch} | Domain {domain} local train accuracy {results_dict["acc"]*100:.2f}%')
        results_dict = train_local_validate(epoch, model, val_dataloader, metric)
        args.log_file.info(f'Epoch: {epoch} | Domain {domain} local validation accuracy {results_dict["acc"]*100:.2f}%')
        if results_dict['acc']>best_val_acc:
            best_val_acc = results_dict['acc']
            best_model = model
            best_epoch = epoch
            SaveCheckPoint(args, model, epoch, os.path.join('saved_model', str(round)+args.dataset+args.test_domain+domain+'_l5.pt'), optimizer=optimzier, note=domain)
    args.log_file.info(f'Best epoch: {best_epoch} | Domain {domain}\'s best local validation accuracy {best_val_acc*100:.2f}%')
    return best_model

def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        z,output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


class _interp_branch(nn.Module):
    '''
    one branch of the interpolator network
    '''

    def __init__(self, in_channels, out_channels):
        super(_interp_branch, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
                                   #nn.ReLU(True),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels*4, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        return self.model(x)
    
def train_local_with_domain_density_translator(round, args, domain, train_dataloader, dataloader, val_dataloader, trans, model, optimizer, criterion, metric):
    best_val_acc = 0.0
    best_epoch = 0
    best_model = None
    for epoch in range(args.local_epochs):    
        model.train()
        for i, data_list in enumerate(train_dataloader):

            imgs, labels, domain_labels = data_list
            imgs = imgs.cuda()
            labels = labels.cuda()
            domain_labels = domain_labels.cuda()
           
            optimizer.zero_grad()
            z,output = model(imgs)
            loss1 = criterion(output, labels)
         
            #z = z.view(z.size(0), -1)
            with torch.no_grad():
                sampled_numbers = random.choices(torch.arange(0, args.n_domains), k=len(imgs))
                y_domain_new = torch.tensor(sampled_numbers)
                y_domain_new = y_domain_new.to(args.device)
                y_domain_onehot = domain_labels.new_zeros([domain_labels.shape[0], args.n_domains])
                y_domain_onehot.scatter_(1, domain_labels[:, None], 1)
                y_domain_new_onehot = y_domain_new.new_zeros([y_domain_new.shape[0], args.n_domains])
                y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
            z_new = trans(z, y_domain_onehot, y_domain_new_onehot)
            reg1 = F.mse_loss(z_new, z)
            '''start = 0
            mid = 0.1
            end = 1.01
            interpol = _interp_branch(z.shape[1], z.shape[1]).to(args.device)
            z_list = []
            samples = torch.arange(start,end,mid)
            iterator = samples
            #iterator = torch.where(samples < 0.5, samples * 0.5, 1 - (1 - samples) * 0.5)
            #iterator = samples * 0.2 + 0.4
            for v in iterator.to(args.device):
                z_list.append(z+v*interpol((z_new-z).reshape(len(imgs),-1, 1, 1)).reshape(len(imgs),-1))
            zi = torch.vstack([ten for ten in z_list])
            perm = torch.randperm(zi.size(0))
            idx1 = perm[:len(imgs)]
            #idx2 = perm[-self.batch_size:]
            zi1 = zi[idx1]
            #zi2 = zi[idx2]
            zj = zi[-len(imgs):]
            reg = args.gamma * F.cross_entropy(model.fc(zi),torch.cat(11*[labels],dim=0))+ args.gamma * F.mse_loss(zj,z_new) + args.alpha * reg1'''
        
            loss = loss1 + args.alpha*reg1 
           
            loss.backward()
            optimizer.step()
            metric.update(output, labels)
            #print(f'Result is : {metric.results()["loss"]*100:.2f}%')
            #print(f'Result is : {metric.results()["loss"]*100:.2f}%')
        results_dict = metric.results()
        args.log_file.info(f'Epoch: {epoch} | Domain {domain}\'s train accuracy after density translation {results_dict["acc"]*100:.2f}%')
        results_dict = train_local_validate(epoch, model, val_dataloader, metric)
        args.log_file.info(f'Epoch: {epoch} | Domain {domain}\'s validation accuracy after density translation {results_dict["acc"]*100:.2f}%')
        if results_dict['acc']>=best_val_acc:
            best_val_acc = results_dict['acc']
            best_model = model
            best_epoch = epoch
            SaveCheckPoint(args, model, epoch, os.path.join('saved_model', str(round)+args.dataset+args.test_domain+domain+'_nn.pt'), optimizer=optimizer, note=domain)
    args.log_file.info(f'Best epoch: {best_epoch} | Domain {domain}\'s validation accuracy after density translation {best_val_acc*100:.2f}%')
    return best_model, best_val_acc 

def site_test(args, epochs, model, dataloader, metric):
    with torch.no_grad():
        for imgs, labels, domains in dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            z,output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict

def run(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    '''dataset and dataloader'''
    dataobjreal = pacsdataset.PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    real_dataloader_dict, real_dataset_dict = dataobjreal.GetData()
    
    domain_list.remove(args.test_domain)           

    criterion = torch.nn.CrossEntropyLoss() 
    metric = Classification()

    '''optimizer dict and local model dict'''
    optim_dict = {}
    net_locals = {}
    '''global model'''
    global_model = models.__dict__[args.arch_name](pretrained=True)
    global_model.fc = torch.nn.Linear(global_model.fc.in_features, args.hidden_dim)
    outer_layer = nn.Linear(args.hidden_dim, args.num_classes)
    fbn = nn.BatchNorm1d(args.hidden_dim, affine=False)
    global_f = f(global_model, nn.Sequential(fbn, outer_layer))
    #global_f = f(global_model,outer_layer)
    global_f = global_f.to(args.device)
    for domain in domain_list:
        net_t = models.__dict__[args.arch_name](pretrained=True)
        net_t.fc = torch.nn.Linear(net_t.fc.in_features, args.hidden_dim)
        outer_layer2 = nn.Linear(args.hidden_dim, args.num_classes)
        fbn2 = nn.BatchNorm1d(args.hidden_dim, affine=False)
        local_f = f(net_t, nn.Sequential(fbn2, outer_layer2))
        #local_f = f(net_t,outer_layer2)
        local_f = local_f.to(args.device)
        net_locals[domain] = local_f
        optim = torch.optim.SGD(local_f.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optim_dict[domain] = optim

    for domain in domain_list: 
        #args.local_epochs = 30
        #best_ckpt = train_local('b', args, domain,  net_locals[domain], optim_dict[domain], real_dataloader_dict[domain]['train'], real_dataloader_dict[domain]['val'], criterion, metric)
        #net_locals[domain].load_state_dict(best_ckpt.state_dict())
        net_locals[domain].load_state_dict(torch.load('saved_model/b{}{}{}_l3.pt'.format(args.dataset,args.test_domain,domain))['model'])

    #Code for generating TSNE plots for z
    '''filtered_dataloader = {}
    for domain in domain_list:
        sampled_indices = []
        flags = [False, False, False, False, False, False, False]
        class_indices = {}
        for target in targets:
            target = int(target)
            if flags[target] == False:
                indices = [i for i, (_, label, domain) in enumerate(real_dataset_dict[domain]['train']) if label == target]
                sampled_index = random.choice(indices)
                sampled_indices.append(sampled_index)
                class_indices[target] = indices
                flags[target] = True
            else:
                sampled_index = random.choice(class_indices[target])
                sampled_indices.append(sampled_index)
        filtered_dataset = Subset(real_dataset_dict[domain]['train'], sampled_indices)
        filtered_dataloader[domain] = DataLoader(filtered_dataset, batch_size=args.bs, shuffle=True)'''
    #Code for generating latent representations
    for version in range(80,args.version):
        targets = torch.LongTensor([random.randint(0, 6) for _ in range(args.bs)]).to('cuda')
        inputs = torch.randn((args.bs, args.hidden_dim), requires_grad=True, device='cuda', dtype=torch.float)
        optimizer = torch.optim.Adam([inputs], lr=args.alr, betas=[0.5, 0.9], eps = 1e-8)
        do_clip = True

        r_loss_layer_dict = {}
        for domain in domain_list:
            for module in net_locals[domain].classifier.modules():
                if isinstance(module, nn.BatchNorm1d):
                    r_loss_layer_dict[domain] = DeepInversionFeatureHook(module)

        markers = np.array(["o" , "v" , "*", "+"])
        colors = np.array([ '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2','#ff7f0e'])
        alpha_values = np.array([0.33333,0.66666,1])
        domainid = 0
        dom_list_x = []
        dom_list_label = []
        dom_list_dom = []
        t_list_x = []
        t_list_label = []
        t_list_dom = []

        for domain in domain_list:
            print(f'Domain: {domain}')
            for iterations in range(10000):
                optimizer.zero_grad()
                outputs = net_locals[domain].classifier(inputs)
                loss_main = criterion(outputs,targets)
                diversity_loss = torch.norm(torch.mm(inputs, inputs.T) - torch.eye(args.bs).cuda())
                l2_loss = torch.norm(inputs.view(args.bs, -1), dim=1).mean()
                
                loss = args.gamma*l2_loss + loss_main + args.beta * r_loss_layer_dict[domain].r_feature 
                loss.backward()
                optimizer.step()
               
            #original_images, original_labels, original_domains = next(iter(filtered_dataloader[domain]))
            #original_images = original_images.cuda()
            #original_labels = original_labels.cuda()
            #final_z = torch.cat([inputs,net_locals[domain].encoder(original_images)])
            #final_labels = torch.cat([targets,original_labels])
            #final_z = torch.cat([inputs,net_locals[domain].encoder(original_images)])
            #final_labels = torch.cat([targets,original_labels])
            
            #final_labels = final_labels.to(torch.int32)
            #labi = final_labels.detach().cpu().numpy()
            #dom_list_label.extend(labi)
            #dom_list_x.extend(final_z.detach().cpu().numpy())
            #dom_list_dom.extend([int(domainid)]*len(final_labels))

            t_list_label.extend(targets.detach().cpu())
            t_list_x.append(inputs.detach().cpu())
            t_list_dom.extend([torch.tensor(domainid, dtype=torch.int)]*len(targets))

            domainid+=1

        #finalz = np.stack(dom_list_x)
        #labi = np.stack(dom_list_label)
        #doms = np.stack(dom_list_dom)

        tfinalz = torch.cat((t_list_x))
        tlabi = torch.stack((t_list_label), dim=0)
        tdoms = torch.stack((t_list_dom), dim=0)
        prefix = 'latents/latent5'
        torch.save((tfinalz, tlabi, tdoms), f'{prefix}_{args.dataset}_{args.test_domain}_{version}.pt')
        args.log_file.info(f'{version}')
        '''marker = torch.cat((torch.zeros(args.bs, dtype=torch.int), torch.ones(args.bs, dtype=torch.int),torch.zeros(args.bs, dtype=torch.int),torch.ones(args.bs, dtype=torch.int),torch.zeros(args.bs, dtype=torch.int), torch.ones(args.bs, dtype=torch.int))).numpy()
        #marker = torch.cat((torch.zeros(64, dtype=torch.int), torch.zeros(64, dtype=torch.int),torch.zeros(64, dtype=torch.int))).numpy()

        pca = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        results = pca.fit_transform(finalz)
        
        scatter, ax =mscatter(results[:,0],results[:,1], ax=None, c=colors[labi], m=markers[marker], alpha=alpha_values[doms])
        all_labels = ['0','1','2','3','4','5','6']
        label_row = all_labels
        label_column = ['Synthetic Z', 'Encoded Z']#,'Interp']
        rows = [mpatches.Patch(color=colors[i]) for i in range(7)]
        columns = [plt.plot([], [], markers[i], markerfacecolor='w', markeredgecolor='k')[0] for i in range(2)]

        #plt.figure()
        #scatter, ax =mscatter(results[:,0],results[:,1], ax=None, c=colors[labi], m=markers[domi.detach().cpu().numpy()])
        plt.legend(rows + columns, label_row + label_column, loc="best", ncol=6, fontsize='xx-small')
    
        plt.savefig('invert_pacs_{}_version_{}.png'.format(args.test_domain,version),bbox_inches='tight')'''
    
    #dataobjreal = fakedataset.PACS_DG(dataset=args.dataset, paths='latents/latent5', test_domain=args.test_domain, batch_size=args.batch_size//2, versions=args.version, version_index=[0])
    #dataloader_dict, dataset_dict = dataobjreal.GetData()
    args.latent_dim = 512
    args.hidden_dim = 1024
    args.source_models = net_locals
    #solver = Solver(dataloader_dict['train'], args, 'l32_{}'.format(args.seed))
    #solver.train()
    trans = load_stargan(args.gan_path + '{}_domain{}_10000-G_l32.ckpt'.format(args.dataset,args.test_domain), args.latent_dim, args.hidden_dim, args.n_domains).to(args.device)
    #trans = load_stargan(args.gan_path + '{}_domain{}_10000-G_l32_{}.ckpt'.format(args.dataset,args.test_domain,args.seed),  args.latent_dim, args.hidden_dim, args.n_domains).to(args.device)
    #trans = load_stargan(args.gan_path + '{}_domain{}_10000-G.ckpt'.format(args.dataset,maps2[args.test_domain])).to(args.device)
    trans.eval()
    val_dict = {}
    args.local_epochs = 20
    best_round_val_acc = 0.0
    for r in range(args.comm):
        FedUpdate(net_locals, global_f)
        '''for domain in domain_list:
            #Train local models
            best_ckpt = train_local(r, args, domain,  net_locals[domain], optim_dict[domain], real_dataloader_dict[domain]['train'], real_dataloader_dict[domain]['val'], criterion, metric)
            net_locals[domain].load_state_dict(best_ckpt.state_dict())'''
        ##### Train local models with domain density model ############
        for domain in domain_list:
            best_ckpt, best_acc_d = train_local_with_domain_density_translator(r, args, domain, real_dataloader_dict[domain]['train'], None, real_dataloader_dict[domain]['val'], trans, net_locals[domain], optim_dict[domain], criterion, metric)
            net_locals[domain].load_state_dict(best_ckpt.state_dict())
        ########## Compute importance weights #################
        weight_dict = Cal_Weight_Dict(real_dataset_dict, site_list=domain_list)
        omega_dictionaries = {}
        for domain in domain_list:
            omega = site_calc_imp(net_locals[domain], optim_dict[domain], real_dataloader_dict[domain]['train'])
            omega_dictionaries[domain] = omega
        omega_new_dictionaries = {}
        omega_new_dictionaries = normalize_dictionaries(omega_dictionaries)
        FedAvg(net_locals, weight_dict, omega_new_dictionaries, global_f)
        #FedAvg2(net_locals, weight_dict, global_f)
        val=0.0
        for domain in domain_list:
            dict  = site_test(args, r, global_f, real_dataloader_dict[domain]['val'], metric)
            val = val+(dict["acc"]*100)
            if domain not in val_dict:
                val_dict[domain] = [dict["acc"]*100]
            else:
                val_dict[domain].append(dict["acc"]*100)   
        dict2  = site_test(args, r, global_f, real_dataloader_dict[args.test_domain]['test'], metric)
        args.log_file.info(f'Round: {r} | Validation Accuracy of Global Model: {val:.2f}%')
        args.log_file.info(f'Round: {r} | Global Federated Model on Real Data Test: {dict2["loss"]:.4f} | Acc: {dict2["acc"]*100:.2f}%')
        if val>best_round_val_acc:
            best_round_val_acc = val
            best_global_model = copy.deepcopy(global_f)
            SaveCheckPoint(args, best_global_model, r, os.path.join('saved_model', args.dataset+args.test_domain+'_globalmod_l.pt'), optimizer=None, note=args.test_domain)
            best_round = r
            best_test_acc = dict2["acc"]*100
    args.log_file.info(f'Round: {best_round} | Best Global Federated Model on Test Data: | Acc: {best_test_acc:.2f}%')
    args.log_file.info(f'Round: {best_round} | Best Global Federated Model on Validation Data: | Acc: {best_round_val_acc:.2f}%')
    for domain in domain_list:
        args.log_file.info(f'Round: {best_round} | Validation Accuracy on : {domain} | Acc: {val_dict[domain][best_round]}%') 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=7)
    parser.add_argument('--hidden_dim', help='size of z', type=int, default=512)
    parser.add_argument('--n_domains', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--seed', type=int, default=20, help='seed')
    parser.add_argument("--test_domain", type=str, default='s', choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    parser.add_argument('--gan_path', type=str, default='saved/stargan_model/')
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=20)
    parser.add_argument('--comm', help='epochs number', type=int, default=20)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--alr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--alpha', help='alpha', type=float, default=10.)
    parser.add_argument('--gamma', help='gamma', type=float, default=0.001)
    parser.add_argument('--beta', help='beta', type=float, default=0.01)
    #parser.add_argument('--latent_dim', help='size of z', type=int, default=256)
    #Deep Inversion parameters
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--setting_id', default=0, type=int, help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='batch size')
    parser.add_argument('--version', default=200, type=int, help='version of generated images')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label for optimization')
    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--dlr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2', help = "arch name from torchvision models to act as a verifier")
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')
    parser.add_argument('--display', help='display in controller', action='store_true')

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=224, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    #parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=bool, default=False)

    # Directories.
    parser.add_argument('--data_dir', type=str, default='../data/')
    #parser2.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='saved/stargan_model')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    
    '''log part'''
    file_name = 'fedavg_'+os.path.split(__file__)[1].replace('.py', '')
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)

    args.log_ten = log_ten
    args.log_file = log_file

    '''setting seed'''
    SEED = int(args.seed)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    run(args)


if __name__ == '__main__':
    main()