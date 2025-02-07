import datetime
import os
import sys
import time
from test import get_style_vectors

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from munch import Munch
from torch.backends import cudnn

from manipulator.checkpoint.checkpoint import CheckpointIO
from manipulator.data.test_dataset import get_test_loader
from manipulator.data.train_dataset import (InputFetcher, get_train_loader, get_val_loader)
from manipulator.models.model import create_model
from manipulator.options.train_options import TrainOptions
from manipulator.util import util
from manipulator.util.logger import StarganV2Logger

sys.path.append(os.getcwd())


def save_checkpoint(epoch):
    for ckptio in ckptios:
        ckptio.save(epoch)


def load_checkpoint(epoch):
    for ckptio in ckptios:
        ckptio.load(epoch)


def reset_grad():
    for optim in optims.values():
        optim.zero_grad()


def get_lr(net, opt):
    if net == 'generator':
        return opt.g_lr
    elif net == 'discriminator':
        return opt.d_lr
    elif net == 'style_encoder':
        return opt.e_lr
    elif net == 'mapping_network':
        return opt.f_lr


def compute_d_loss(nets, opt, x_real, y_org, y_trg, z_trg=None, x_ref=None, x_gt=None, x_gt_mask=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    # x_real.requires_grad_()

    if opt.paired and x_gt is not None:
        x_gt_ = x_gt[x_gt_mask>0]
        if x_gt_.shape[0] > 0:
            y_trg_ = y_trg[x_gt_mask>0]
            out = nets.discriminator(x_gt_, y_trg_)
            loss_real = adv_loss(out, 1)
        else:
            loss_real = x_real.new_zeros(1)
    else:
        out = nets.discriminator(x_real, y_org)
        loss_real = adv_loss(out, 1)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref)
        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item())


def compute_g_loss(nets, opt, x_real, y_org, y_trg, z_trg=None, x_ref=None, x_gt=None, x_gt_mask=None):
    assert (z_trg is None) != (x_ref is None)
    # adversarial loss
    if z_trg is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # the first param corresponds to the jaw opening (similar to lip distance)
    dist_real = x_real[:, :, 0]
    dist_fake = x_fake[:, :, 0]

    # mouth loss (Pearson Correlation Coefficient)
    v_real = dist_real - torch.mean(dist_real, dim=1, keepdim=True)
    v_fake = dist_fake - torch.mean(dist_fake, dim=1, keepdim=True)
    loss_mouth_f = 1 - torch.mean(torch.mean(v_real * v_fake, dim=1) * torch.rsqrt(
        torch.mean(v_real ** 2, dim=1)+1e-7) * torch.rsqrt(torch.mean(v_fake ** 2, dim=1)+1e-7))

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg)) * opt.lambda_sty

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real)) * opt.lambda_cyc

    # mouth loss backward
    dist_rec = x_rec[:, :, 0]

    v_rec = dist_rec - torch.mean(dist_rec, dim=1, keepdim=True)
    loss_mouth_b = 1 - torch.mean(torch.mean(v_fake * v_rec, dim=1) * torch.rsqrt(
        torch.mean(v_fake ** 2, dim=1)+1e-7) * torch.rsqrt(torch.mean(v_rec ** 2, dim=1)+1e-7))

    loss_mouth = (loss_mouth_f + loss_mouth_b) * opt.lambda_mouth

    if opt.paired and x_gt is not None and x_gt_mask.sum() > 0:
        loss_paired = (F.l1_loss(x_fake[:, :, 1:], x_gt[:, :, 1:], reduction='none') * x_gt_mask.view(
            -1, 1, 1)).sum(0).mean() / x_gt_mask.sum() * opt.lambda_paired
    else:
        loss_paired = x_real.new_zeros(1)

    loss = loss_adv + loss_sty + loss_cyc + loss_mouth + loss_paired
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       cyc=loss_cyc.item(),
                       mouth=loss_mouth.item(),
                       paired=loss_paired.item())


def adv_loss(logits, target):
    """Implements LSGAN loss"""
    assert target in [1, 0]
    return torch.mean((logits - target)**2)


@torch.no_grad()
def generate_exp_coeffs(nets, opt, mode='latent'):
    assert opt.test_dir is not None

    if mode == 'latent':
        opt.trg_emotions = opt.selected_emotions
        opt.ref_dirs = None
    elif mode == 'reference':
        opt.trg_emotions = None
        opt.ref_dirs = opt.ref_dirs or ['celebrities/reference/Nicholson',
                                        'celebrities/reference/Pacino', 'celebrities/reference/DeNiro']
    else:
        raise NotImplementedError
    opt.batch_size = 1
    num_exp_coeffs = 51

    loader_src = get_test_loader(os.path.join(opt.test_dir, 'DECA'), opt)
    loaders_ref = [get_test_loader(os.path.join(dir, 'DECA'), opt)
                   for dir in opt.ref_dirs] if opt.ref_dirs is not None else None

    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    # calculate style vector(s)
    s_refs = get_style_vectors(nets, opt, loader_src, loaders_ref)

    # translate and return
    with torch.no_grad():
        all_outputs = []
        for s_ref in s_refs:  # for all emotions
            gaussian = cv2.getGaussianKernel(opt.seq_len, -1)
            gaussian = torch.from_numpy(gaussian).float().to(
                device).repeat(1, num_exp_coeffs)
            output = torch.zeros(len(loader_src)+opt.seq_len-1,
                                 num_exp_coeffs).float().to(device)

            original = torch.zeros(
                len(loader_src)+opt.seq_len-1, num_exp_coeffs).float().to(device)

            for i, (x_src, _) in enumerate(loader_src):
                # Prepare input sequence.
                x_src = x_src.to(device)

                # Translate sequence.
                x_fake = nets.generator(x_src, s_ref)

                output[i:i+opt.seq_len] = output[i:i+opt.seq_len] + \
                    torch.squeeze(x_fake, dim=0)*gaussian
                original[i:i+opt.seq_len] = torch.squeeze(x_src, dim=0)

            for i, (x_src, _) in enumerate(loader_src):
                if i == 0:
                    # Prepare input sequence.
                    x_src = x_src.to(device)
                    # Translate sequence.
                    x_fake = nets.generator(x_src, s_ref)
                    output[:opt.seq_len-1] = torch.squeeze(x_fake, dim=0)[:-1]
                if i == len(loader_src)-1:
                    # Prepare input sequence.
                    x_src = x_src.to(device)
                    # Translate sequence.
                    x_fake = nets.generator(x_src, s_ref)
                    output[-opt.seq_len+1:] = torch.squeeze(x_fake, dim=0)[1:]

            output = output.cpu().numpy()
            all_outputs.append(output)

            original = original.cpu().numpy()

    return np.stack(all_outputs), original


if __name__ == '__main__':

    opt = TrainOptions().parse()
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # initialize train dataset
    loader_src = get_train_loader(opt, which='source')
    loader_ref = get_train_loader(opt, which='reference')

    # initialize val dataset
    loader_src_val = get_val_loader(opt, which='source')

    # initialize models
    nets = create_model(opt)

    # print network params and initialize them
    for name, module in nets.items():
        util.print_network(module, name)
        print('Initializing %s...' % name)
        module.apply(util.he_init)

    # set optimizers
    optims = Munch()
    for net in nets.keys():
        optims[net] = torch.optim.Adam(params=nets[net].parameters(
        ), lr=get_lr(net, opt), betas=[opt.beta1, opt.beta2])

    ckptios = [CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_nets.pth'), opt, len(opt.gpu_ids) > 0, **nets),
               CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_optims.pth'), opt, False, **optims)]

    # create logger
    logger = StarganV2Logger(opt.checkpoints_dir)

    # Training loop
    if opt.finetune:
        # load nets if finetuning
        ckptios[0].load(opt.finetune_epoch)
        for ckptio in ckptios:
            ckptio.fname_template = ckptio.fname_template.replace(
                '.pth', '_finetuned.pth')
    else:
        # resume training if necessary
        if opt.resume_epoch > 0:
            load_checkpoint(opt.resume_epoch)

    loss_log = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
    logfile = open(loss_log, "a")

    # Learning rate cache for decaying.
    lr = optims.generator.param_groups[0]['lr']
    #f_lr = optims.mapping_network.param_groups[0]['lr']

    fetcher = InputFetcher(loader_src, loader_ref, opt.latent_dim)
    fetcher_val = InputFetcher(loader_src_val, loader_ref, opt.latent_dim)

    print('Start training...')
    start_time = time.time()
    for epoch in range(0 if opt.finetune else opt.resume_epoch, opt.niter):
        for model in nets:
            nets[model].train()

        for i in range(len(loader_src)):

            # fetch sequences and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, y_trg = inputs.x_ref, inputs.y_ref
            z_trg = inputs.z_trg
            x_gt, x_gt_mask = inputs.x_gt, inputs.x_gt_mask

            x_real = x_real.to(device)
            y_org = y_org.to(device)
            x_ref = x_ref.to(device)
            y_trg = y_trg.to(device)
            z_trg = z_trg.to(device)
            x_gt = x_gt.to(device)
            x_gt_mask = x_gt_mask.to(device)

            # train the discriminator
            # d_loss, d_losses_latent = compute_d_loss(nets, opt, x_real, y_org, y_trg, z_trg=z_trg)
            # reset_grad()
            # d_loss.backward()
            # optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_gt=x_gt, x_gt_mask=x_gt_mask)
            reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator (and F, E)
            # g_loss, g_losses_latent = compute_g_loss(nets, opt, x_real, y_org, y_trg, z_trg=z_trg, x_gt=x_gt, x_gt_mask=x_gt_mask)
            # reset_grad()
            # g_loss.backward()
            # optims.generator.step()
            # optims.mapping_network.step()
            # optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_gt=x_gt, x_gt_mask=x_gt_mask)
            reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()

            iteration = i + epoch*len(loader_src)
            # print out log info
            if (i+1) % opt.print_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Epoch [%i/%i], Iteration [%i/%i], " % (
                    elapsed, epoch+1, opt.niter, i+1, len(loader_src))
                all_losses = dict()
                for loss, prefix in zip([d_losses_ref, g_losses_ref],  # d_losses_latent, g_losses_latent
                                        ['D/ref_', 'G/ref_']):  # 'D/latent_', 'G/latent_',
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                        logger.log_training(
                            "train/"+prefix+key, value, iteration)
                log += ' '.join(['%s: [%.4f]' % (key, value)
                                for key, value in all_losses.items()])

                print(log)
                logfile.write(log)
                logfile.write('\n')

        # ----------------------- calculate total eval loss on the validation set ----------------------- #
        if (epoch + 1) % opt.val_freq == 0:
            with torch.no_grad():
                for model in nets:
                    nets[model].eval()
                all_losses = dict()
                for i in range(len(loader_src_val)):
                    # fetch sequences and labels
                    inputs = next(fetcher_val)
                    x_real, y_org = inputs.x_src, inputs.y_src
                    x_ref, y_trg = inputs.x_ref, inputs.y_ref
                    z_trg = inputs.z_trg
                    x_gt = inputs.x_gt
                    x_gt_mask = inputs.x_gt_mask

                    x_real = x_real.to(device)
                    y_org = y_org.to(device)
                    x_ref = x_ref.to(device)
                    y_trg = y_trg.to(device)
                    z_trg = z_trg.to(device)
                    x_gt = x_gt.to(device)
                    x_gt_mask = x_gt_mask.to(device)

                    # get discriminator losses
                    # d_loss, d_losses_latent = compute_d_loss(
                    #     nets, opt, x_real, y_org, y_trg, z_trg=z_trg)
                    d_loss, d_losses_ref = compute_d_loss(
                        nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_gt=x_gt, x_gt_mask=x_gt_mask)

                    # get generator losses (and F, E)
                    # g_loss, g_losses_latent = compute_g_loss(
                    #     nets, opt, x_real, y_org, y_trg, z_trg=z_trg)
                    g_loss, g_losses_ref = compute_g_loss(
                        nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_gt=x_gt, x_gt_mask=x_gt_mask)

                    # print out log info
                    for loss, prefix in zip([d_losses_ref, g_losses_ref],  # d_losses_latent, g_losses_latent
                                            ['D/ref_', 'G/ref_']):  # 'D/latent_', 'G/latent_',
                        for key, value in loss.items():
                            k = prefix + key
                            if k not in all_losses:
                                all_losses[k] = 0
                            else:
                                all_losses[k] += value*x_real.size(0)
                # print(all_losses)
                for (key, value) in all_losses.items():
                    # get mean across all samples
                    all_losses[key] = all_losses[key]/len(loader_src.dataset)
                    logger.log_training("val/"+key, all_losses[key], epoch)

                log = "Validation, Epoch [%i/%i] " % (epoch+1, opt.niter)

                log += ' '.join(['%s: [%.4f]' % (key, value)
                                for key, value in all_losses.items()])

                print(log)
                logfile.write(log)
                logfile.write('\n')

        # ----------------------- generate some example meshes and add to tensorboard ----------------------- #
        # exp_coeffs, exp_coeffs_ground = generate_exp_coeffs(nets, opt, mode='latent')

        # shape, shape_detail_images, tex_images, tex_detail_images = generate_mesh(exp_coeffs_ground, opt)
        # logger.log_mesh(shape,'shape/original',epoch)
        # logger.log_mesh(shape_detail_images,'shape_detail/original',epoch)
        # logger.log_mesh(tex_images,'tex_images/original',epoch)
        # logger.log_mesh(tex_detail_images,'tex_detail/original',epoch)

        # for k, e in enumerate(exp_coeffs):
        #     shape, shape_detail_images, tex_images, tex_detail_images = generate_mesh(e, opt)
        #     logger.log_mesh(shape, 'shape/{}'.format(opt.selected_emotions[k]), epoch)
        #     logger.log_mesh(shape_detail_images, 'shape_detail/{}'.format(opt.selected_emotions[k]), epoch)
        #     logger.log_mesh(tex_images, 'tex_images/{}'.format(opt.selected_emotions[k]), epoch)
        #     logger.log_mesh(tex_detail_images, 'tex_detail/{}'.format(opt.selected_emotions[k]), epoch)

        # exp_coeffs, _ = generate_exp_coeffs(nets, opt, mode='reference')

        # for k, e in enumerate(exp_coeffs):
        #     ref_name = opt.ref_dirs[k]
        #     if ref_name.endswith('/'):
        #         ref_name = ref_name[:-1]
        #     ref_name = ref_name.rsplit('/', 1)[-1]
        #     shape, shape_detail_images, tex_images, tex_detail_images = generate_mesh(e, opt)
        #     logger.log_mesh(shape, 'shape/Reference {}'.format(ref_name), epoch)
        #     logger.log_mesh(shape_detail_images, 'shape_detail/Reference {}'.format(ref_name), epoch)
        #     logger.log_mesh(tex_images, 'tex_images/Reference {}'.format(ref_name), epoch)
        #     logger.log_mesh(tex_detail_images, 'tex_detail/Reference {}'.format(ref_name), epoch)

        # save model checkpoints
        if (epoch+1) % opt.save_epoch_freq == 0:
            save_checkpoint(epoch=epoch+1)

        # Decay learning rates.
        if (epoch+1) > (opt.niter - opt.niter_decay) and epoch != opt.niter - 1:
            log = 'Decayed learning rate'
            for net in nets.keys():
                lr_new = optims[net].param_groups[0]['lr'] - \
                    (get_lr(net, opt) / float(opt.niter_decay))
                for param_group in optims[net].param_groups:
                    param_group['lr'] = lr_new
                log += f', {net} {lr_new:g}'

            print(log)
            logfile.write(log)
            logfile.write('\n')

    logfile.close()
