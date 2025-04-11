'''
This file was adapted from the original MMC github repository.
Authors: Ekdeep Singh Lubana et al.
Date: 09.04.2025
GitHub: https://github.com/EkdeepSLubana/MMC
Paper: https://arxiv.org/pdf/2211.08422
Modifcation: Modified probe_connect to integrate in our codebase
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

torch.manual_seed(int(0))
cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from tqdm import tqdm
import numpy as np
from models.ResNet import create_model
from mmc_utils import linear_eval
from utils.utils import get_data_loaders



### Linear path interpolation
def lmc(alpha, model_1, model_2, model_class, num_classes):
    torch.cuda.empty_cache()
    net_interpolated = create_model(model_class, num_classes).to(device)

    for module_1, module_2, module_interpolated in zip(model_1.modules(), model_2.modules(), net_interpolated.modules()):
        if(isinstance(module_1, nn.Conv2d)):
            module_interpolated.weight.data = alpha * module_1.weight.data + (1-alpha) * module_2.weight.data
        elif(isinstance(module_1, nn.BatchNorm2d)):
            module_interpolated.weight.data = alpha * module_1.weight.data + (1-alpha) * module_2.weight.data
            module_interpolated.bias.data = alpha * module_1.bias.data + (1-alpha) * module_2.bias.data
            module_interpolated.running_mean.data = alpha * module_1.running_mean.data + (1-alpha) * module_2.running_mean.data
            module_interpolated.running_var.data = alpha * module_1.running_var.data + (1-alpha) * module_2.running_var.data
        elif(isinstance(module_1, nn.Linear)):
            module_interpolated.weight.data = alpha * module_1.weight.data + (1-alpha) * module_2.weight.data
            module_interpolated.bias.data = alpha * module_1.bias.data + (1-alpha) * module_2.bias.data
        else:
            pass
        
    return net_interpolated


### Quadratic path interpolation
def quadmc(alpha, model_1, model_2, net_midpoint, model_class, num_classes):
    torch.cuda.empty_cache()
    net_interpolated = create_model(model_class, num_classes).to(device)

    for module_1, module_2, module_midpoint, module_interpolated in zip(model_1.modules(), model_2.modules(), net_midpoint.modules(), net_interpolated.modules()):
        if(isinstance(module_1, nn.Conv2d)):
            module_interpolated.weight.data = (alpha**2) * module_1.weight.data + 2 * alpha * (1-alpha) * module_midpoint.weight.data + ((1-alpha)**2) * module_2.weight.data
        elif(isinstance(module_1, nn.BatchNorm2d)):
            module_interpolated.weight.data = (alpha**2) * module_1.weight.data + 2 * alpha * (1-alpha) * module_midpoint.weight.data + ((1-alpha)**2) * module_2.weight.data
            module_interpolated.bias.data = (alpha**2) * module_1.bias.data + 2 * alpha * (1-alpha) * module_midpoint.bias.data + ((1-alpha)**2) * module_2.bias.data
            module_interpolated.running_mean.data = (alpha**2) * module_1.running_mean.data + 2 * alpha * (1-alpha) * module_midpoint.running_mean.data + ((1-alpha)**2) * module_2.running_mean.data
            module_interpolated.running_var.data = (alpha**2) * module_1.running_var.data + 2 * alpha * (1-alpha) * module_midpoint.running_var.data + ((1-alpha)**2) * module_2.running_var.data
        elif(isinstance(module_1, nn.Linear)):
            module_interpolated.weight.data = (alpha**2) * module_1.weight.data + 2 * alpha * (1-alpha) * module_midpoint.weight.data + ((1-alpha)**2) * module_2.weight.data
            module_interpolated.bias.data = (alpha**2) * module_1.bias.data + 2 * alpha * (1-alpha) * module_midpoint.bias.data + ((1-alpha)**2) * module_2.bias.data
        else:
            pass
        
    return net_interpolated



### Training quadratic path's midpoint
def train_quad_midpoint(model_1, model_2, train_setup):

    criterion = nn.CrossEntropyLoss()

    # midpoint model
    net_midpoint = create_model(train_setup['model'], num_classes=train_setup['num_classes'], normal_conv_layer=True).to(device)

    # interpolated model used for training
    net_interpolated = create_model(train_setup['model'], num_classes=train_setup['num_classes'], normal_conv_layer=True).to(device)
    net_interpolated.train()

    # optimizer
    optimizer = optim.SGD(net_midpoint.parameters(), lr=0.0, momentum=0.9, weight_decay=1e-4)

    # training 
    dataloader, _ = get_data_loaders(train_setup["data_config"])
    accs_midpoint = []
    for epoch in range(train_setup['n_epochs']):
        torch.cuda.empty_cache()

        base_lr = train_setup['start_lr'] * (train_setup['lr_decay'] ** (epoch // train_setup['decay_epochs']))
        
        train_loss, correct, total = 0, 0, 0
        with tqdm(dataloader, unit=" batch") as tepoch:
            batch_idx = 0
            for inputs, targets in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}")
                batch_idx += 1
                
                # Quad interpolation
                with torch.no_grad():
                    # interpolation weight
                    alpha = torch.rand(1)[0]

                    # update optimizer LR 
                    optimizer.param_groups[0]['lr'] = base_lr * (2 * alpha * (1-alpha))

                    # interpolate model params (has to be done explicitly because BN parameters will be overriden)
                    for module_1, module_2, module_midpoint, module_interpolated in zip(model_1.modules(), model_2.modules(), net_midpoint.modules(), net_interpolated.modules()):
                        if(isinstance(module_1, nn.Conv2d)):
                            module_interpolated.weight.data = (alpha**2) * module_1.weight.data + 2 * alpha * (1-alpha) * module_midpoint.weight.data + ((1-alpha)**2) * module_2.weight.data
                        elif(isinstance(module_1, nn.BatchNorm2d)):
                            module_interpolated.weight.data = (alpha**2) * module_1.weight.data + 2 * alpha * (1-alpha) * module_midpoint.weight.data + ((1-alpha)**2) * module_2.weight.data
                            module_interpolated.bias.data = (alpha**2) * module_1.bias.data + 2 * alpha * (1-alpha) * module_midpoint.bias.data + ((1-alpha)**2) * module_2.bias.data
                        elif(isinstance(module_1, nn.Linear)):
                            module_interpolated.weight.data = (alpha**2) * module_1.weight.data + 2 * alpha * (1-alpha) * module_midpoint.weight.data + ((1-alpha)**2) * module_2.weight.data
                            module_interpolated.bias.data = (alpha**2) * module_1.bias.data + 2 * alpha * (1-alpha) * module_midpoint.bias.data + ((1-alpha)**2) * module_2.bias.data
                        else:
                            pass

                # get gradients for interpolated model
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net_interpolated(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # associate grads from interpolated model to intermediate model
                with torch.no_grad():
                    for module_1, module_2, module_midpoint, module_interpolated in zip(model_1.modules(), model_2.modules(), net_midpoint.modules(), net_interpolated.modules()):
                        if(isinstance(module_1, nn.Conv2d)):
                            module_midpoint.weight.grad = module_interpolated.weight.grad.data
                        elif(isinstance(module_1, nn.BatchNorm2d)):
                            module_midpoint.weight.grad = module_interpolated.weight.grad.data
                            module_midpoint.bias.grad = module_interpolated.bias.grad.data
                        elif(isinstance(module_1, nn.Linear)):
                            module_midpoint.weight.grad = module_interpolated.weight.grad.data
                            module_midpoint.bias.grad = module_interpolated.bias.grad.data
                        else:
                            pass

                # update intermediate model
                optimizer.step()
                        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                tepoch.set_postfix(loss=train_loss/batch_idx, accuracy=100. * correct/total, lr=optimizer.param_groups[0]['lr'], base_lr=base_lr)

        accs_midpoint += [100. * correct/total]

    return net_midpoint, accs_midpoint



### Probe connectivity
def probe_connect(model_1, model_2, net_midpoint=None, setup=None, return_loss=False):

    np.random.seed(setup['seed_id'])

    # trainloader: used for resetting BN
    train_loader, test_loader = get_data_loaders(setup["data_config"])

    #get subset of train set for loss landscape eval
    idx = torch.randperm(len(train_loader.dataset))[:5000]
    test_loader = DataLoader(Subset(train_loader.dataset, idx), batch_size=setup["data_config"]["batch_size"])   

    # eval accuracy
    accs_list, loss_list = [], []
    print("Interpolation strengths:", np.arange(0, 1 + 1 / setup['n_interpolations'], 1 / setup['n_interpolations']))
    for alp in np.arange(0, 1 + 1 / setup['n_interpolations'], 1 / setup['n_interpolations']):
        torch.cuda.empty_cache()

        # interpolate model
        if(setup['connect_pattern']=='LMC' or setup['connect_pattern']=='LMCP'):
            net_interpolated = lmc(alpha=alp, model_1=model_1, model_2=model_2, model_class=setup['model_class'], num_classes=setup['num_classes'])
        elif(setup['connect_pattern']=='QMC'):
            net_interpolated = quadmc(alpha=alp, model_1=model_1, model_2=model_2, net_midpoint=net_midpoint, model_class=setup['model_class'], num_classes=setup['num_classes'])
        else:
            raise Exception("MC not implemented")
        
        # Update BN stats 
        # Technically few samples should be enough, but momentum is very small for computing running stats and hence we use full dataloader
        net_interpolated.train()
        with torch.no_grad():
            n_samples = 0
            for x, _ in train_loader:
                n_samples += x.shape[0]
                net_interpolated(x.to(device))

        # eval
        if return_loss:
            #a, l = linear_eval(net_interpolated.eval(), dataloader=train_loader, suffix='for alpha: '+str(alp), return_loss=return_loss)
            a, l = linear_eval(net_interpolated.eval(), dataloader=test_loader, suffix='for alpha: '+str(alp), return_loss=return_loss)
            loss_list += [l]
        else:
            a = linear_eval(net_interpolated.eval(), dataloader=test_loader, suffix='for alpha: '+str(alp))

        accs_list += [a]

    if return_loss:
        return accs_list, loss_list
    else:
        return accs_list