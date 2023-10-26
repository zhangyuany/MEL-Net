try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from numpy import unravel_index


plt.switch_backend('agg')
import numpy as np

from collections import defaultdict
from math import log10
from statistics import mean
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import Data, ToTensor, RandomHorizontalFlip
from aspp_unet import Unet
#from pykalman import KalmanFilter
import os
os.environ['CUDA_ENABLE_DEVICES'] = '0'
import scipy.io
from scipy.io import savemat


def split_dataset(dataset, test_percentage=0.1):#对数据集进行切片，取10%为测试集，其余的为验证集
    """
    Split a dataset in a train and test set.

    Parameters
    ----------
    dataset : dataset.Data
        Custom dataset object.
    test_percentage : float, optional
        Percentage of the data to be assigned to the test set.
    """
    test_size = round(len(dataset) * test_percentage)#测试的数据等于数据集乘测试的百分比
    train_size = len(dataset) - test_size#训练数据的多少是数据集的多少减去测试的
    return random_split(dataset, [train_size, test_size])









def iter_epoch(
        models, optimizers, dataset, device='cuda:0', batch_size=64,
        eval=False, reconstruction_criterion=nn.MSELoss(),#改了，原来是MSE
        use_fk_loss=True):

    """
    Train both generator and discriminator for a single epoch.
    Parameters
    ----------
    G : torch.nn.Module
        Generator  models respectively.
    optim_G : torch.optim.Optimizer
        Optimizers for both the models. Using Adam is recommended.
    train_dataloader : torch.utils.data.Dataloader
        Dataloader of real images to train the discriminator on.真实图像的加载器，用于判别器的训练
    device : str, optional
        Device to train the models on.用于训练模型的设备
    batch_size : int, optional
        Number of samples per batch.
    eval : bool, optional
        If `True`, model parameters are not updated
                    batch_size=64, eval=False,
    reconstruction_criterion: loss used to evaluate the reconstruction quality 用于评估重建质量的损失
        options: nn.MSELoss(), nn.L1Loss(), None (if used, only GAN loss is
        counted)
    is_fk_loss: bool
        If 'True', loss is evaluated in the fk space, else loss is evaluated
        directly
   fk:频域信息
    Returns
    -------
    dict
        Dictionary containing the mean loss values for the generator  and the mean PSNR .
    """



    def update_generator(lores_batch, hires_batch,maxx):
        """Update the generator over a single minibatch."""

        if eval:
            G.eval()
        else:
            G.train()

        # Generate superresolution and transform.
        sures_batch = G(lores_batch)
       # sures_batch=torch.Tensor().long()
        #print(type(sures_batch))
        #sures_batch=sures_batch.to(device=device,dtype=torch.int64)
        #hires_batch = hires_batch.to(device=device, dtype=torch.int64)
        #lores_batch = lores_batch.to(device=device, dtype=torch.int64)
        #lores:输入；hires:标签；sures:输出

        if use_fk_loss:
           hires_fk_batch = transform_fk(#干净的标签
               hires_batch, output_dim, is_batch=True)
           sures_fk_batch = transform_fk(#网络学习后的输出
               sures_batch, output_dim, is_batch=True)

        # Initialize losses.
        rec_loss = 0
        rec_fk_loss = 0
        regularization_loss = 0
        #for name, param in G.named_parameters():
           # if 'conv.weight' in name:
             #   regularization_loss += torch.sum(abs(param))
       # print(regularization_loss)

        #regularization_loss = 0
        for name , param in G.named_parameters():
            if 'conv.weight'in name:
                regularization_loss +=torch.sum(abs(param))
           # print(param)
          #  regularization_loss += torch.sum(abs(param))
            #tensor = tensor.to(device=torch.device("cuda:0"))
            #lamda =torch.tensor(10**-1)
           # l1_alpha=0.0001

       # def l1_regularization(models, l1_alpha):
           # l1_loss = []
           # for module in models.modules:
               # if type(module) is nn.BatchNorm2d:
                  #  l1_loss.append(torch.abs(module.weight).sum())
           # return l1_alpha * sum(l1_loss)

        if content_criterion is not None:
            rec_loss = content_criterion(sures_batch, hires_batch) #学习到的图与干净的图比较，震源

         '''
         带fk的是指频域信息，在震源定位中没有用上
         '''
            if use_fk_loss:
                #rec_fk_loss = nn.functional.mse_loss(sures_fk_batch, hires_fk_batch)
                rec_fk_loss = content_criterion(sures_fk_batch, hires_fk_batch)




        loss_G = 1 * rec_loss  # + 1*rec_fk_loss

        if not eval:
            #loss_G.backward(retain_graph = True)
            loss_G.backward()
            optim_G.step()
            optim_G.zero_grad()
        data_transforms = transforms.Compose([
            #  RandomHorizontalFlip(),
            ToTensor()
        ])
        dataset = Data(
            args.filename_x, args.filename_y, args.data_root,
            transform=data_transforms, flag1=0)

        #psnr111 = 10 * np.log10(maxx**2/ (np.mean(np.square((hires_batch.cpu().numpy() - sures_batch.cpu().detach().numpy())))))



        psnr = 10 * log10(maxx**2 / nn.functional.mse_loss(
        sures_batch, hires_batch).item())
        hires = hires_batch.cpu().numpy()
        sures = sures_batch.detach().cpu().numpy()
        ps = np.sum(np.square(hires))
        pn = np.sum(np.square(hires - sures))
        psnr111 = 10 * np.log10(ps / pn)
        # true = np.where(hires == np.max(hires))
        #
        # pred = np.where(sures == np.max(sures))
        # max_target=np.max(np.max(np.max(hires,0),1),1)
        #
        # max_output = np.max(np.max(np.max(sures, 0), 1), 1)

        max_target = np.max(hires, axis=(2, 3))
        max_output = np.max(sures,axis=(2,3))


        # true_x = np.argmax(hires)
        #计算的是预测震源位置与实际震源位置之间的距离
        true1= unravel_index(hires[0,:,:,:].argmax(),shape=hires.shape)
        pred1 = unravel_index(sures[0,:,:,:].argmax(), sures.shape)
        true_x1=true1[3]
        true_z1 = true1[2]
        pred_x1 = pred1[3]
        pred_z1 = pred1[2]
        wucha1 = (true_x1 - pred_x1) * (true_x1 - pred_x1) + (true_z1 - pred_z1) * (true_z1 - pred_z1)

        true2 = unravel_index(hires[1, :, :, :].argmax(), shape=hires.shape)
        pred2 = unravel_index(sures[1, :, :, :].argmax(), sures.shape)
        true_x2 = true2[3]
        true_z2 = true2[2]
        pred_x2 = pred2[3]
        pred_z2 = pred2[2]
        wucha2 = (true_x2 - pred_x2) * (true_x2 - pred_x2) + (true_z2 - pred_z2) * (true_z2 - pred_z2)

        # true3 = unravel_index(hires[2, :, :, :].argmax(), shape=hires.shape)
        # pred3 = unravel_index(sures[2, :, :, :].argmax(), sures.shape)
        # true_x3 = true3[3]
        # true_z3 = true3[2]
        # pred_x3 = pred3[3]
        # pred_z3 = pred3[2]
        # wucha3 = (true_x3 - pred_x3) * (true_x3 - pred_x3) + (true_z3 - pred_z3) * (true_z3 - pred_z3)
        #
        # true4 = unravel_index(hires[3, :, :, :].argmax(), shape=hires.shape)
        # pred4 = unravel_index(sures[3, :, :, :].argmax(), sures.shape)
        # true_x4 = true4[3]
        # true_z4 = true4[2]
        # pred_x4 = pred4[3]
        # pred_z4 = pred4[2]
        # wucha4 = (true_x4 - pred_x4) * (true_x4 - pred_x4) + (true_z4 - pred_z4) * (true_z4 - pred_z4)
        #
        # true5 = unravel_index(hires[4, :, :, :].argmax(), shape=hires.shape)
        # pred5 = unravel_index(sures[4, :, :, :].argmax(), sures.shape)
        # true_x5 = true5[3]
        # true_z5 = true5[2]
        # pred_x5 = pred5[3]
        # pred_z5 = pred5[2]
        # wucha5 = (true_x5 - pred_x5) * (true_x5 - pred_x5) + (true_z5 - pred_z5) * (true_z5 - pred_z5)
        #
        # true6 = unravel_index(hires[5, :, :, :].argmax(), shape=hires.shape)
        # pred6 = unravel_index(sures[5, :, :, :].argmax(), sures.shape)
        # true_x6 = true6[3]
        # true_z6 = true6[2]
        # pred_x6 = pred6[3]
        # pred_z6 = pred6[2]
        # wucha6 = (true_x6 - pred_x6) * (true_x6 - pred_x6) + (true_z6 - pred_z6) * (true_z6 - pred_z6)

        wucha=(wucha1+wucha2)*25
        # +wucha3+wucha4+wucha5+wucha6)*25






            # wuchaa=np.append(wucha)
        # print('true',true)
        # print('pred',true[3])
        # return wucha

        # true_z = np.argmax(np.argmax(hires, axis=3), axis=2)
        # pred_x=np.argmax(sures,axis=3)
        # print('pred_max:', pred_x.shape)
        # print('true_max:',true_x)

        # truee=[]



        # true_x = np.where(np.where(hires==max_target,axis=2)==max_target,axis=2)
        # true_x = np.where(np.isclose(hires,max_target,atol=1e-4))
        # true_z = np.where(hires[2] == max_target)
        #
        # pred_x = np.where(sures[3] == max_output)
        # pred_z = np.where(sures == max_output)

        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # print('pred:',pred_z)
        # print('true_x:', wucha)
        # print('pred_z:', pred_x)





        # hires = hires_batch.detach().cpu().numpy()
        # sures = sures_batch.detach().cpu().numpy()
        # true = np.where(hires == np.max(hires))
        # true_x=true[3]
        # true_z=true[2]
        # print('true:',np.max(hires))
        #
        # pred = np.where(sures == np.max(sures))
        # pred_x = pred[0][3]
        # pred_z = pred[2]
        # print('pred:',pred_x)
        # wucha=(true_x-pred_x)*(true_x-pred_x)+(true_z-pred_z)*(true_z-pred_z)
        # wucha1=(np.sum(wucha) / 8)*25
        # wucha=np.sum(np.square(true[3]-pred[3]))+np.sum(np.square(true[2]-pred[2]))








        error=pow(wucha/2,0.5)








       ### return loss_G.item(), psnr, psnr111#, ssim.item()
        return loss_G.item(),psnr,  psnr111 ,error#

    G = models
    optim_G = optimizers


    output_dim = dataset[0]['y'].shape[1:]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=(not eval), shuffle=False)

    #mean_loss =[]
    mean_loss_G = []
    mean_psnr = []
    mean_psnr111 = []
    mean_error=[]
    #mean_ssim = []

    content_criterion = reconstruction_criterion

    for sample in dataloader:
        lores_batch = sample['x'].to(device).float()
        hires_batch = sample['y'].to(device).float()
        xx=sample['max'].numpy()
        maxx = np.sum(xx)/8
        # true = np.where(hires_batch == np.max(hires))





        #sigma = np.max(abs(x)) * 0.1
        #w = np.random.normal(0, sigma, size=(91, 150, 150))
       # w=torch.from_numpy(w).to(device).float()
        #lores_batch1 = lores_batch+w

        #ssim to add
        #loss_G, psnr, psnr111 = update_generator(lores_batch, hires_batch)
       # loss_G, psnr111 = update_generator(lores_batch, hires_batch)
        loss_G, psnr ,psnr111,error= update_generator(lores_batch, hires_batch,maxx)#这里被我改了

        mean_loss_G.append(loss_G)

        mean_psnr.append(psnr)
        mean_psnr111.append(psnr111)
        mean_error.append(error)
        #mean_loss.append(loss)
        #mean_loss.append(loss)
        #mean_ssim.append(ssim)

    return {
        'G': mean(mean_loss_G),
        'psnr': mean(mean_psnr),
        'psnr111': mean(mean_psnr111),
        'error':mean(mean_error)
       # 'ssim': mean(mean_ssim)
    }


# def transform_fk(image, dataset_dim, is_batch=False):
#     """
#     Apply the Fourier transform of an image (or batch of images) and
#     compute the magnitude of its real and imaginary parts.
#     """#应用图像或一批图像的傅立叶变换
#     if not is_batch:
#         image = image.unsqueeze(0)
#
#     image = torch.nn.functional.interpolate(image, size=dataset_dim)
#     image_fk = torch.rfft(image, 2, normalized=True)
#     image_fk = image_fk.pow(2).sum(-1).sqrt()
#
#     return image_fk
def transform_fk(image, dataset_dim, is_batch=False):
    """
    Apply the Fourier transform of an image (or batch of images) and
    compute the magnitude of its real and imaginary parts.
    """#应用图像或一批图像的傅立叶变换
    if not is_batch:
        image = image.unsqueeze(0)

    image = torch.nn.functional.interpolate(image, size=dataset_dim)
    #image_fk=torch.rfft(image,2,normalize=true)
    #image_fk = torch.fft.fft2(image, dim=(-2,-1))  #做了修改
    image_fk = rfft(image, 2)
    image_fk = image_fk.pow(2).sum(-1).sqrt()

    return image_fk


def plot_samples(generator, dataset, epoch, device='cuda', directory='image',
                 is_train=False):
    """
    Plot data samples, their superresolution and the corresponding fk
    transforms.
这个是画什么的：输入，地震数据
    """
    def add_subplot(plt, image, i, idx, title=None, cmap='viridis'):
        plt.subplot(num_rows, num_cols, num_cols * idx + i)

        if idx == 0:
            plt.title(title)

        plt.imshow(image.squeeze().detach().cpu(),
                   interpolation='none', cmap=cmap)
        plt.axis('off')

    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    sample = next(iter(dataloader))

    lores_batch = sample['x'].to(device).float()
    hires_batch = sample['y'].to(device).float()

    generator.eval()

    sures_batch = generator(lores_batch)
    # true = np.where(hires_batch == np.max(hires_batch))[0]
    # pred = np.where(sures_batch == np.max(sures_batch))[0]
    # print('true:', true)
    # print('pred:', pred)


    num_cols = 3
    num_rows = 2
    output_dim = dataset[0]['y'].shape[1:]

    plt.figure(figsize=(6, 6))

    for idx, (lores, sures, hires) \
            in enumerate(zip(lores_batch, sures_batch, hires_batch)):
        # Plot images.


        add_subplot(plt, lores, 1, idx, "Input ",cmap='gray')#cmap指的是映射
        add_subplot(plt, sures, 2, idx, "Output",cmap='gray')
        add_subplot(plt, hires, 3, idx, "clear",cmap='gray')
        # true = np.where(hires == np.max(hires))
        # pred = np.where(sures == np.max(sures))
        # print('true:', true)
        # print('pred:', pred)
        hh = hires.squeeze(0)

        # Plot transformed images.
        add_subplot(plt, transform_fk(lores, output_dim), 4, idx, "In fk")
        add_subplot(plt, transform_fk(sures, output_dim), 5, idx, "In fk")
        add_subplot(plt, transform_fk(hires, output_dim), 6, idx, "In fk")
        #sures=sures.squeeze()
        sures=sures.detach().cpu( ).numpy()
        scipy.io.savemat('results/result_20/shuchu.mat',{'data':sures})
        lores = lores.detach().cpu().numpy()
        scipy.io.savemat('results/result_20/shuru.mat', {'data': lores})
        hires = hires.detach().cpu().numpy()
        scipy.io.savemat('results/result_20/biaoqian.mat', {'data': hires})



    plt.tight_layout()
    if not is_train:#如果没被训练过，是val
        plt.savefig(os.path.join(directory, f'samples_val_{epoch:03d}.pdf'))


    else:#被训练过的放在train
        plt.savefig(os.path.join(directory, f'samples_train_{epoch:03d}.pdf'))
    plt.close()





def save_loss_plot(loss_g, directory, is_val=False, name=None):
    plt.figure()
    plt.plot(loss_g, label="Loss")
    plt.legend()
    if is_val:
        if name is None:
            plt.savefig(f"{directory}/loss_val.png")
        else:
            plt.savefig(f"{directory}/loss_val_{name}.png")
    else:
        if name is None:
            plt.savefig(f"{directory}/loss.png")
        else:
            plt.savefig(f"{directory}/loss_{name}.png")

    plt.close()


'''
这一块是做什么的：保存使用的参数，txt文件
'''
def main(args):
    # Create directories if it's not  hyper-optimisation round.
    if not args.is_optimisation:
        results_directory = f'results/result_{args.experiment_num}'
        os.makedirs('images', exist_ok=True)
        os.makedirs(results_directory, exist_ok=True)
        # Save arguments for experiment reproducibility.保存实验可重复性参数
        with open(os.path.join(results_directory, 'arguments.txt'), 'w') \
                as file:
            json.dump(args.__dict__, file, indent=2)

    # Set size for plots.画什么？
    plt.rcParams['figure.figsize'] = (10, 10)

    # Select the device to train the model on.
    device = torch.device(args.device)

    # Load the dataset.
    # TODO : Add normalisation  transforms.Normalize(
    #   torch.tensor(-4.4713e-07).float(),
    #   torch.tensor(0.1018).float())
    # TODO: Add more data augmentation transforms.
    data_transforms = transforms.Compose([
      #  RandomHorizontalFlip(),
        ToTensor()
    ])

    dataset = Data(
        args.filename_x, args.filename_y, args.data_root,
        transform=data_transforms, flag1=0)



    if not args.is_optimisation:
        print(f"Data sizes, input: {dataset.input_dim}, output: "
              f"{dataset.output_dim}, Fk: {dataset.output_dim_fk}")

    train_data, test_data = split_dataset(dataset, args.test_percentage +  args.val_percentage )
    test_data, val_data = split_dataset(test_data, 0.5 )



    if args.model == 'Unet':

       generator = Unet(1).to(device)


    # Optimizers优化程序

    optim_G = optim.Adam(generator.parameters(), lr=args.lr)
    print(optim_G.state_dict()['param_groups'][0]['lr'])

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_G, patience=args.scheduler_patience,factor=0.5, verbose=True)





    # losses type
    criterion_dictionary = {
        "MSE": nn.MSELoss(),
        "L1": nn.L1Loss(),       #是不是把L1改成卡尔曼，作为损失函数的约束项？
        "Entroy":nn.BCELoss(),
        #"MAE":nn.MAELoss(),
       # "L2":nn.L2Loss(),
    }
    reconstruction_criterion = criterion_dictionary[args.criterion_type]

    # Initialize a dict of empty lists for plotting.
    plot_log = defaultdict(list)

    for epoch in range(args.n_epochs):
        # Train model for one epoch.
        loss = iter_epoch(
            (generator),
            (optim_G), train_data, device,
            batch_size=args.batch_size,
            reconstruction_criterion=reconstruction_criterion,
            use_fk_loss=args.use_fk_loss)
        psnr =iter_epoch(
            (generator),
            (optim_G),train_data, device,
            batch_size=args.batch_size,
            reconstruction_criterion=reconstruction_criterion)

        # Report model performance.
        if not args.is_optimisation:
            print(f"Epoch: {epoch}, Loss: {loss['G']}, "
                  f"PSNR: {loss['psnr']} "
                 # f"PSNR111: {loss['psnr111']} ")

                  f"SNR: {loss['psnr111']},"
                  f"error:{loss['error']} ")
            with open(os.path.join(results_directory,'result.txt'),'a+')as f:
                f.write(f"Epoch: {epoch}, Loss: {loss['G']},"
                        f"PSNR: {loss['psnr']} "
                        f"error: {loss['error']} "
                        f"SNR: {loss['psnr111']}\n")

        plot_log['G'].append(loss['G'])
        #lr_list=[]
        #lr_list.append(optim_G.state_dict()['param_groups'][0]['lr'])
        print(optim_G.state_dict()['param_groups'][0]['lr'])
       # with open(os.path.join(results_directory, 'result.txt'), 'a+')as f:
       #     f.write(optim_G.state_dict()['param_groups'][0]['lr'])
        #np.savetxt('lr.txt',lr_list,fmt="%f")
        #with open('lr.txt')as F:
           # for line in F:
              #  print(line,end='')

       # plot_log['G'].append(loss['psnr'])


        # Model evaluation every eval_iteration and last iteration.
        if epoch % args.eval_interval == 0 \
                or (args.is_optimisation and epoch == args.n_epochs - 1):
            loss_val = iter_epoch(
                (generator),
                (None), val_data, device,
                batch_size=args.batch_size, eval=True,
                reconstruction_criterion=reconstruction_criterion ,
                use_fk_loss=args.use_fk_loss)
            if not args.is_optimisation:
                print(f"Validation on epoch: {epoch}, Lossval: {loss_val['G']}, "
                      f" PSNRval: {loss_val['psnr']},"
                       f" errorval: {loss_val['error']},"
                      f"SNRval: {loss_val['psnr111']}")
                with open(os.path.join(results_directory, 'result.txt'), 'a+')as f:
                    f.write(f"Validation on epoch: {epoch}, Lossval: {loss_val['G']}, "
                            f" PSNRval: {loss_val['psnr']},"
                             f" errorval: {loss_val['error']},"
                            f"SNRval: {loss_val['psnr111']}\n")
                      #f"#, SSIM: {loss_val['ssim']}")

            plot_log['G_val'].append(loss_val['G'])
           # plot_log['psnr_val'].append(loss_val['psnr111'])
            #plot_log['psnr_val'].append(loss_val['psnr'])
           # plot_log['ssim_val'].append(loss_val['ssim'])

            # Update scheduler based on PSNR or separate model losses.
            if args.is_psnr_step:
                scheduler_g.step(loss_val['psnr'])


            else:
                scheduler_g.step(loss_val['G'])

            if not args.is_optimisation:
                pass
                # save_loss_plot(plot_log['G_val'], results_directory, is_val=True)

        if not args.is_optimisation:
            # Plot results.
            if epoch % args.save_interval == 0:
                plot_samples(generator, val_data, epoch, device,
                             results_directory)
                plot_samples(generator, train_data, epoch, device,
                             results_directory, is_train=True)

            save_loss_plot(plot_log['G'], results_directory)
       # plt.plot(range(args.n_epochs),lr_list, color='r')
           # save_psnr_plot(plot_log['G'], results_directory)

    if not args.is_optimisation:
        # Save the trained generator model.
        torch.save(generator, os.path.join(results_directory, 'generator.pth'))
        torch.save(generator.state_dict(), os.path.join(results_directory, 'param_generator.pth'))

        if args.save_test_dataset:
            sets_name = ['test', 'val', 'train']
            sets = [test_data, val_data, train_data]
            for name, d_set in zip(sets_name, sets):
                list_x = []
                list_y = []
                for sample in d_set:
                    list_x.append(sample['x'].unsqueeze(0))
                    list_y.append(sample['y'].unsqueeze(0))
                tensor_x = torch.cat(list_x, 0)
                tensor_y = torch.cat(list_y, 0)
                data_folder_for_results = 'final/data'
                os.makedirs(data_folder_for_results, exist_ok=True)
                torch.save(tensor_x, f'{data_folder_for_results}/{name}_data_x_{args.experiment_num}.pt')
                torch.save(tensor_y, f'{data_folder_for_results}/{name}_data_y_{args.experiment_num}.pt')

        return plot_log, generator, test_data
    if args.is_optimisation:
        __, test_data = random_split(test_data, [len(test_data)-2, 2])
        return plot_log, generator, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a superresolution model for reducing spatial "
        "aliasing in seismic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data arguments.
    data_group = parser.add_argument_group('Data')

    data_group.add_argument(
        '--data_root', '-d', type=str, default='train_data',
        help="Root directory of the data.")
    data_group.add_argument(
       # '--filename_x', '-x', type=str, default='smooth_shot_six288',
        '--filename_x', '-x', type=str, default='smooth4_rand_inputnoise',
       # '--filename_x', '-x', type=str, default='vz_rand_inputnoise',
        help="Name of the low resolution data file (without the '.mat' "
        "extension).")
    data_group.add_argument(
       # '--filename_y', '-y', type=str, default='vz_rand_labelnoise',
        '--filename_y', '-y', type=str, default='smooth4_rand_labelnoise',
       #  '--filename_y', '-y', type=str, default='smooth_label_six288',
        help="Name of the high resolution data filee (without the '.mat' "
        "extension).")
    data_group.add_argument(
        '--test_percentage', type=float, default=0.1,
        help="Size of the test set")
    data_group.add_argument(
        '--val_percentage', type=float, default=0.1,
        help="Size of the test set")
    data_group.add_argument(
        '--filename_out', '-o', type=str, default='blind_test_original',
        #'--filename_out', '-o', type=str, default='noisy',
        help="Name of the high resolution data filee (without the '.mat' "
             "extension).")
#
    data_group.add_argument("--model_folder", type=str, default="test_data",
                        help="Folder with the model to generate SR from")#里面装的是测试的东西

    # Model arguments.
    model_group = parser.add_argument_group('Model')

    model_group.add_argument(
        '--model', type=str, default="Unet",

        help="Model type.")
    model_group.add_argument(
        '--latent_dim', type=int, default=256,
        help = "dimensionality of the latent space, only relevant for Unet")
    model_group.add_argument(
        '--num_res_blocks', type=int, default=4,
        help="Number of resblocks in model, only relevant for Unet")

    # Training arguments.
    training_group = parser.add_argument_group('Training')

    training_group.add_argument(
        '--n_epochs', type=int, default=201,
        help="number of epochs")
    training_group.add_argument(
        '--batch_size', type=int, default=8,
        help="batch size")
    training_group.add_argument(
        '--lr', type=float, default=0.01,#，这里是0。01
        help="learning rate")
        #help='initial learning rate for Adam')
    training_group.add_argument(
        '--scheduler_patience', type=int, default="3",#5个epoch中val不下降，降低学习率
        help="How many val epochs of no improvement to consider Plateau")
    training_group.add_argument(
        '--is_psnr_step', type=int, default="0",#这里本来是0，现在被改了
        help="Use PSNR for scheduler or separate losses")
    training_group.add_argument(
         '--criterion_type', type=str, default="MSE",
         choices=['MSE', 'L1', 'Entroy','None'],
         help="Reconstruction criterion to use.")
    training_group.add_argument(
        '--use_fk_loss', type=int, default="0",
        help="Use loss in fk space or not, 0 for False and 1 for True")#使用傅立叶域的变化

    # Misc arguments.
    misc_group = parser.add_argument_group('Miscellaneous')

    misc_group.add_argument(
        '--eval_interval', type=int, default=4,
        help="evaluate on test set every eval_interval epochs")
    misc_group.add_argument(
        '--save_interval', type=int, default=4,
        help="Save images every SAVE_INTERVAL epochs")
    misc_group.add_argument(
        '--device', type=str, default="cuda:0",
        help="Training device 'cpu' or 'cuda:0'")
    misc_group.add_argument(
        '--experiment_num', '-n', type=int, default=44,)
    misc_group.add_argument(
        "--is_optimisation", type=int, default=0,
        help="True or False for whether the run is called by the hyperopt"#运行时是否参数优化
    )
    misc_group.add_argument(
        "--save_test_dataset", type=int, default=1,
        help="True or False for option to save test dataset "
    )

    args = parser.parse_args()

    main(args)


