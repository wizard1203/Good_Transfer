# mkdir checkpoints
# cd checkpoints
# wget https://download.pytorch.org/models/resnet50-19c8e357.pth


import numpy as np
import customized_resnet as models
import argparse
import os
import torch
from myDataLoader import myDataLoader
import time
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from skip import skip

from efficient_loader import efficient_load_partition_data_cifar10
from resnet_v2 import ResNet18


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./test_img',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='pretrained_model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--pretrained', default=['./checkpoints/resnet50-19c8e357.pth'],
                    help='path to res pretrained checkpoint')
parser.add_argument('--output_dir', default=['./result/supervised_DIP/'])

# parser.add_argument('--pretrained', default=['./checkpoints/resnet50-19c8e357.pth',
#                                              './checkpoints/moco_v2.pth'],
#                     help='path to res pretrained checkpoint')
# parser.add_argument('--output_dir', default=['./result/supervised_DIP/',
#                                              './result/moco_v2'])

parser.add_argument('--which_layer', default='layer4')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--initial_size', default=256, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--max_iter', default=3000, type=int)
parser.add_argument('--ckpt_iter', default=[1000,3000,5000])


def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Make dir: %s'%dir)

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def get_noise(input_depth, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input

def postp(tensor): # to clip results in the range [0,1]
    postpa = transforms.Compose([transforms.Normalize(
                                     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.
    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    #print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        #print(param.data.shape)
        own_state[name].copy_(param.data)


def main():
    args = parser.parse_args()
    print("=> creating pre-trained model '{}'".format(args.arch))

    # for dir in args.output_dir:
    #     checkdir(dir)

    save_img_dir = os.path.join("/home/comp/20481896/", "datasets/cifar10")
    save_model_dir= os.path.join("/home/comp/20481896/", "DecoupleGradVariance/experiments/standalone/DecoupleGradVariance/checkpoints")
    target_path = os.path.join(save_model_dir, "partition_alpha=0.1checkpoint-server-standalone-FedAvg-resnet18_v2-cifar10-128-0.1-no-hetero-0.1-False-epoch-999-default.pth")
    args.output_dir = save_model_dir
    checkdir(args.output_dir)

    # pretrained_model = [models.__dict__[args.arch]() for path in args.pretrained]
    pretrained_model = ResNet18(None, 10)
    ckp_T = torch.load(target_path)
    load_my_state_dict(pretrained_model, ckp_T)

    # load from pre-trained, before DistributedDataParallel constructor
    # for pi,path in enumerate(args.pretrained):
    #     if os.path.isfile(path):
    #         print("=> loading checkpoint '{}'".format(path))
    #         checkpoint = torch.load(path, map_location="cpu")

    #         try:
    #             state_dict = checkpoint['model']
    #         except:
    #             try:
    #                 state_dict = checkpoint['state_dict']
    #             except:
    #                 state_dict = checkpoint
    #         for k in list(state_dict.keys()):
    #             if k.startswith('fc.'):
    #                 del state_dict[k]
    #             elif k.startswith('module.fc.'):
    #                 del state_dict[k]
    #             elif k.startswith('module'):
    #                 state_dict[k.replace('module.','')]=state_dict[k]
    #                 del state_dict[k]

    #         args.start_epoch = 0
    #         msg = pretrained_model[pi].load_state_dict(state_dict, strict=False)
    #         print(msg.missing_keys)
    #         if len(msg.missing_keys):
    #             assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    #         print("=> loaded pre-trained model '{}'".format(path))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(path))
    #     pretrained_model[pi].cuda().eval()

    # for pi in range(len(args.pretrained)):
    #     for name, param in pretrained_model[pi].named_parameters():
    #         param.requires_grad = False

    pretrained_model.cuda().eval()
    for name, param in pretrained_model.named_parameters():
        param.requires_grad = False

    # dir = os.path.join(args.data, 'val')
    # dataset = myDataLoader(dir)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers)

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = efficient_load_partition_data_cifar10(
        "cifar10",
        save_img_dir,
        "hetero",
        0.1,
        10,
        128,
        0,
    )
    mal_client_data = train_data_local_dict[0]

    criterion = nn.MSELoss().cuda()
    criterion2 = TVLoss().cuda()
    input_depth = 32
    imsize_net = 256
    # imsize = 224
    imsize = 32

    # # for i, (img, target, filename) in enumerate(mal_client_data):
    # for i, (img, target) in enumerate(mal_client_data):
    #     # if 1:
    #     if i > 0:
    #         # measure data loading time
    #         img = img.cuda()
            
    #         # for pi in range(len(args.pretrained)):
    #         targets = pretrained_model(img, name=args.which_layer).detach()
    #         for img_i in range(img.size()[0]):
    #             # if 1:
    #             if i > 10:
    #                 # out_path = os.path.join(args.output_dir, args.which_layer, filename[img_i])
    #                 real_path = os.path.join(args.output_dir, args.which_layer, f"RealIMG{[img_i]}.png")
    #                 out_path = os.path.join(args.output_dir, args.which_layer, f"reconIMG{[img_i]}.png")

    #                 real_img = postp(img[img_i].data.cpu().squeeze())
    #                 real_img.save(real_path)
    #                 if not os.path.exists(out_path):
    #                     print('%d-%d'%(i,img_i))
    #                     start=time.time()
    #                     pad = 'zero'  # 'refection'
    #                     net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
    #                                 num_channels_up=[16, 32, 64, 128, 128, 128],
    #                                 num_channels_skip=[4, 4, 4, 4, 4, 4],
    #                                 filter_size_down=[7, 7, 5, 5, 3, 3], filter_size_up=[7, 7, 5, 5, 3, 3],
    #                                 upsample_mode='nearest', downsample_mode='avg',
    #                                 need_sigmoid=False, pad=pad, act_fun='LeakyReLU').type(img.type())

    #                     net_input = get_noise(input_depth, imsize_net).type(img.type()).detach()
    #                     out = net(net_input)[:, :, :imsize, :imsize]
    #                     print(out.size())

    #                     # Compute number of parameters
    #                     s = sum(np.prod(list(p.size())) for p in net.parameters())
    #                     print('Number of params: %d' % s)

    #                     target = targets[[img_i,],...]

    #                     # run style transfer
    #                     max_iter = args.max_iter
    #                     show_iter = 50
    #                     optimizer = optim.Adam(get_params('net', net, net_input), lr=args.lr)
    #                     n_iter = [0]

    #                     while n_iter[0] <= max_iter:

    #                         def closure():
    #                             optimizer.zero_grad()
    #                             out = pretrained_model(net(net_input)[:, :, :imsize, :imsize], name=args.which_layer)
    #                             loss = criterion(out, target) #+ criterion2(net_input)*1e-2
    #                             loss.backward()
    #                             n_iter[0] += 1
    #                             # print loss
    #                             if n_iter[0] % show_iter == (show_iter - 1):
    #                                 print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
    #                             return loss

    #                         optimizer.step(closure)
    #                     out_img = postp(net(net_input)[:, :, :imsize, :imsize].data[0].cpu().squeeze())
    #                     # plt.imshow(out_img)
    #                     # plt.show()
    #                     end = time.time()
    #                     print('Time:'+str(end-start))

    #                     checkdir(os.path.dirname(out_path))
    #                     out_img.save(out_path)


    # for batch_i, (img, target, filename) in enumerate(mal_client_data):
    for batch_i, (img, target) in enumerate(mal_client_data):
        # if 1:
        if batch_i < 3:
            # measure data loading time
            img = img.cuda()
            
            # for pi in range(len(args.pretrained)):
            targets = pretrained_model(img, name=args.which_layer).detach()
            # out_path = os.path.join(args.output_dir, args.which_layer, filename[batch_i])
            out_path = os.path.join(args.output_dir, args.which_layer, f"reconIMG{[batch_i]}-mean.png")

            for img_i in range(5):
                real_path = os.path.join(args.output_dir, args.which_layer, f"RealIMG{[batch_i]}-{img_i}.png")
                real_img = postp(img[img_i].data.cpu().squeeze())
                real_img.save(real_path)

            real_path = os.path.join(args.output_dir, args.which_layer, f"RealIMG{[batch_i]}-mean.png")
            real_img = postp(img.mean(dim=0).data.cpu().squeeze())
            real_img.save(real_path)

            target = targets.mean(dim=0)
            # noised_targets = [target]
            DP_degree = [0, 0.001, 0.01, 0.1, 0.5]
            for i, std in enumerate(DP_degree):
                # noised_targets.append[target + torch.normal(mean=torch.ones(target.shape)*0.0, std=std)]
                if std > 0.0:
                    noised_target = target + torch.normal(mean=torch.ones(target.shape)*0.0, std=std).cuda()
                else:
                    noised_target = target + 0.0
                out_path = os.path.join(args.output_dir, args.which_layer, f"reconIMG{[batch_i]}-mean-DP{std}.png")
                if not os.path.exists(out_path):
                    print('%d-%d'%(batch_i, batch_i))
                    start=time.time()
                    pad = 'zero'  # 'refection'
                    net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
                                num_channels_up=[16, 32, 64, 128, 128, 128],
                                num_channels_skip=[4, 4, 4, 4, 4, 4],
                                filter_size_down=[7, 7, 5, 5, 3, 3], filter_size_up=[7, 7, 5, 5, 3, 3],
                                upsample_mode='nearest', downsample_mode='avg',
                                need_sigmoid=False, pad=pad, act_fun='LeakyReLU').type(img.type())

                    net_input = get_noise(input_depth, imsize_net).type(img.type()).detach()
                    out = net(net_input)[:, :, :imsize, :imsize]
                    print(out.size())

                    # Compute number of parameters
                    s = sum(np.prod(list(p.size())) for p in net.parameters())
                    print('Number of params: %d' % s)


                    # run style transfer
                    max_iter = args.max_iter
                    show_iter = 50
                    optimizer = optim.Adam(get_params('net', net, net_input), lr=args.lr)
                    n_iter = [0]

                    while n_iter[0] <= max_iter:

                        def closure():
                            optimizer.zero_grad()
                            out = pretrained_model(net(net_input)[:, :, :imsize, :imsize], name=args.which_layer)
                            loss = criterion(out, noised_target) #+ criterion2(net_input)*1e-2
                            loss.backward()
                            n_iter[0] += 1
                            # print loss
                            if n_iter[0] % show_iter == (show_iter - 1):
                                print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                            return loss

                        optimizer.step(closure)
                    out_img = postp(net(net_input)[:, :, :imsize, :imsize].data[0].cpu().squeeze())
                    # plt.imshow(out_img)
                    # plt.show()
                    end = time.time()
                    print('Time:'+str(end-start))

                    checkdir(os.path.dirname(out_path))
                    out_img.save(out_path)


if __name__ == '__main__':
    main()                            
                            