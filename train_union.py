from renderer.util.visualizer import Visualizer
from train_options import TrainOptions
from renderer.models.head2head_model import create_model as create_renderer_model
from manipulator.data import train_dataset as mead_dataset
from manipulator.util.logger import StarganV2Logger
from manipulator.util import util as mutil
from manipulator.models.model import create_model as create_manipulator_model
from manipulator.checkpoint.checkpoint import CheckpointIO
from DECA.decalib.datasets.detectors import FAN
from DECA.decalib.utils import util as deca_util
from DECA.decalib.utils.config import cfg as deca_cfg
from DECA.decalib.deca import DECA
from renderer.util import util as render_util
import union_dataset
import datetime
import os
import cv2
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from munch import Munch
from torch.backends import cudnn
from skimage.io import imread
from skimage.transform import warp, estimate_transform

sys.path.append(os.getcwd())


class DECADecoder:
    def __init__(self, crop_size=224, scale=1.25, device=torch.device('cuda')):
        deca_cfg.model.use_tex = True
        self.deca = DECA(config=deca_cfg, device=device)
        self.crop_size = crop_size
        self.scale = scale
        self.face_detector = FAN(device=device)

    def decode(self, out, codedict, imagepath):
        codedict['exp'][0] = out[-1, 1:]
        #codedict['pose'][0, 3] = out[-1, 0]
        opdict, visdict = self.deca.decode(codedict)
        image = np.array(imread(imagepath))
        h, w, _ = image.shape
        bbox, _ = self.face_detector.run(image)
        if len(bbox) < 4:
            print('no face detected! run original image')
            left = 0
            right = h-1
            top = 0
            bottom = w-1
        else:
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]

        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0,
                          bottom - (bottom - top) / 2.0])
        size = int(old_size*self.scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] -
                           size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array(
            [[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        nmfc_image = warp(deca_util.tensor2image(visdict['nmfcs'][0])/255,
                          tform, output_shape=(h, w))
        shape_image = warp(deca_util.tensor2image(visdict['shape_detail_images'][0])/255,
                           tform, output_shape=(h, w))
        return nmfc_image, shape_image


def save_checkpoint(epoch):
    for ckptio in ckptios:
        ckptio.save(epoch)

    export_ckpt(epoch)


def export_ckpt(epoch):
    save_dir = os.path.join(opt.checkpoints_dir, 'export')
    os.makedirs(save_dir, exist_ok=True)
    manipulator_weights = {
        'generator': nets['generator'].module.state_dict(),
        'style_encoder': nets['style_encoder'].module.state_dict(),
        'discriminator': nets['discriminator'].module.state_dict()
    }
    torch.save(manipulator_weights, os.path.join(
        save_dir, f'{epoch:02d}_nets.pth'))
    render_G_weight = nets['rendererG'].module.netG.state_dict()
    torch.save(render_G_weight, os.path.join(
        save_dir, f'{epoch:02d}_net_G.pth'))
    render_D_weight = nets['rendererD'].module.netD.state_dict()
    torch.save(render_D_weight, os.path.join(
        save_dir, f'{epoch:02d}_net_D.pth'))
    if not opt.no_mouth_D:
        render_Dm_weight = nets['rendererD'].module.netDm.state_dict()
        torch.save(render_Dm_weight, os.path.join(
            save_dir, f'{epoch:02d}_net_Dm.pth'))
    if opt.use_eyes_D:
        render_De_weight = nets['rendererD'].module.netDe.state_dict()
        torch.save(render_De_weight, os.path.join(
            save_dir, f'{epoch:02d}_net_De.pth'))


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
    elif net == 'rendererG':
        return opt.rg_lr
    elif net == 'rendererD':
        return opt.rd_lr


def compute_d_loss(nets, opt, x_real, y_org, y_trg, x_ref=None, x_tgt=None, tgt_mask=None):
    out = nets.discriminator(x_real, y_org)
    if x_tgt is not None:
        x_tgt_ = x_tgt[tgt_mask > 0]
        if x_tgt_.shape[0] > 0:
            y_trg_ = y_trg[tgt_mask > 0]
            out = nets.discriminator(x_tgt_, y_trg_)
            loss_real = adv_loss(out, 1)
        else:
            loss_real = x_real.new_zeros(1)
        loss_real = (loss_real + adv_loss(out, 1)) * 0.5
    else:
        loss_real = adv_loss(out, 1)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x_ref)
        x_fake = nets.generator(x_real, s_trg)
        if x_tgt is not None and x_tgt_.shape[0] > 0:
            s_trg_ = s_trg[tgt_mask > 0]
            x_fake_ = nets.generator(x_tgt_, s_trg_)
            x_fake = torch.cat([x_fake, x_fake_], dim=0)
            y_trg = torch.cat([y_trg, y_trg_], dim=0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item())


def compute_g_loss(nets, opt, x_real, y_org, y_trg, x_ref=None, x_tgt=None, tgt_mask=None):
    # adversarial loss
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

    if x_tgt is not None and tgt_mask.sum() > 0:
        m = tgt_mask > 0
        x_tgt_ = x_tgt[m]
        s_tgt = nets.style_encoder(x_tgt_)
        loss_sty_ = torch.mean(torch.abs(s_pred[m] - s_tgt) + torch.abs(s_trg[m] - s_tgt)) / 2 * opt.lambda_sty
        loss_sty = (loss_sty + loss_sty_) * 0.5

        x_tgt_rec = nets.generator(x_tgt_, s_tgt)
        loss_cyc_ = torch.mean(torch.abs(x_tgt_rec - x_tgt_)) * opt.lambda_cyc
        loss_cyc = (loss_cyc + loss_cyc_) * 0.5

        e_real = x_tgt[:, :, 1:]
        e_fake = x_fake[:, :, 1:]
        loss_paired = (1 - (tgt_mask.view(-1, 1) * (e_real * e_fake).mean(-1) * torch.rsqrt(
            e_real.pow(2).mean(-1)+1e-7) * torch.rsqrt(e_fake.pow(2).mean(-1)+1e-7)).sum(0).mean() / tgt_mask.sum()) * opt.lambda_paired
    else:
        loss_paired = x_real.new_zeros(1)

    loss = loss_adv + loss_sty + loss_cyc + loss_mouth + loss_paired
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       cyc=loss_cyc.item(),
                       mouth=loss_mouth.item(),
                       paired=loss_paired.item()), x_fake.detach()


def adv_loss(logits, target):
    """Implements LSGAN loss"""
    assert target in [1, 0]
    return torch.mean((logits - target)**2)


def _compute_renderer_loss(nets, opt, nmfc_video, rgb_video, mouth_centers, eyes_centers, eye_video, shape_video, mask_video, fake_B_last, device, display=False, src_img=None):
    t_len = opt.n_frames_G
    _, h, w = nmfc_video.shape
    # nmfc_video have 3 channels
    nmfc_video = nmfc_video.to(device).view(-1, t_len, 3, h, w)
    input_A = nmfc_video
    input_B = rgb_video.to(device).view(-1, t_len, 3, h,
                                        w)  # rgb_video has 3 channels
    mouth_centers = mouth_centers.to(
        device).view(-1, t_len, 2) if not opt.no_mouth_D else None
    eyes_centers = eyes_centers.to(
        device).view(-1, t_len, 2) if opt.use_eyes_D else None

    if not opt.no_eye_gaze:
        # eye_gaze_video has 3 channels
        eye_gaze_video = eye_video.to(device).view(-1, t_len, 3, h, w)
        input_A = torch.cat([nmfc_video, eye_gaze_video], dim=2)

    if opt.use_shapes:
        # shape_video has 3 channels
        shape_video = shape_video.to(device).view(-1, t_len, 3, h, w)
        input_A = torch.cat([input_A, shape_video], dim=2)

    # mask_video has 3 channels but we keep 1
    mask_video = mask_video.to(device).view(-1, t_len, 3, h, w)
    mask_video = mask_video[:, :, 0:1, :, :]
    input_A = torch.cat([input_A, mask_video], dim=2)

    ############## Forward Pass ######################
    # Identity Embedder and Generator
    fake_B, real_A, real_Bp, fake_B_last = nets.rendererG(
        input_A, input_B, fake_B_last)

    real_B = real_Bp[:, 1:]
    if mouth_centers is not None:
        mouth_centers = mouth_centers[:, t_len-1:, :].contiguous().view(-1, 2)
    if eyes_centers is not None:
        eyes_centers = eyes_centers[:, t_len-1:, :].contiguous().view(-1, 2)
    tensor_list = render_util.reshape([real_B, fake_B, real_A])

    # Image and Mouth, Eyes Discriminators
    losses = nets.rendererD(tensor_list, mouth_centers, eyes_centers)
    losses = [torch.mean(x) if x is not None else 0 for x in losses]
    loss_dict = dict(zip(rendererD.module.loss_names, losses))

    # Losses
    loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
    #loss_dict['G_paired'] = (F.l1_loss(fake_B, real_B.detach(), reduction='none') * mask_video).mean([0,1,2]).sum() / mask_video.sum() * opt.lambda_paired_renderer
    loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] # + loss_dict['G_paired']
    if not opt.no_mouth_D:
        loss_G += loss_dict['Gm_GAN'] + loss_dict['Gm_GAN_Feat']
        loss_D += (loss_dict['Dm_fake'] + loss_dict['Dm_real']) * 0.5
    if opt.use_eyes_D:
        loss_G += loss_dict['Ge_GAN'] + loss_dict['Ge_GAN_Feat']
        loss_D += (loss_dict['De_fake'] + loss_dict['De_real']) * 0.5

    if display:
        visual_dict = [('input_nmfc_image', render_util.tensor2im(nmfc_video[0, -1], normalize=False)),
                       ('fake_image', render_util.tensor2im(fake_B[0, -1])),
                       ('fake_first_image', render_util.tensor2im(
                           fake_B_last[0, -1])),
                       ('real_image', render_util.tensor2im(real_B[0, -1])),
                       ('input_mask_image', render_util.tensor2im(mask_video[0, -1], normalize=False))]
        if src_img is not None:
            visual_dict += [('src_image', cv2.imread(src_img)[:, :, ::-1])]
        if opt.use_shapes:
            visual_dict += [('input_shape_image',
                             render_util.tensor2im(shape_video[0, -1], normalize=False))]
        if not opt.no_eye_gaze:
            visual_dict += [('input_eye_gaze_image',
                             render_util.tensor2im(eye_gaze_video[0, -1], normalize=False))]
        if not opt.no_mouth_D:
            mc = render_util.fit_ROI_in_frame(
                mouth_centers[-1], opt)
            fake_B_mouth = render_util.tensor2im(
                render_util.crop_ROI(fake_B[0, -1], mc, opt.ROI_size))
            visual_dict += [('fake_image_mouth', fake_B_mouth)]
            real_B_mouth = render_util.tensor2im(
                render_util.crop_ROI(real_B[0, -1], mc, opt.ROI_size))
            visual_dict += [('real_image_mouth', real_B_mouth)]
        if opt.use_eyes_D:
            mc = render_util.fit_ROI_in_frame(
                eyes_centers[-1], opt)
            fake_B_eyes = render_util.tensor2im(
                render_util.crop_ROI(fake_B[0, -1], mc, opt.ROI_size))
            visual_dict += [('fake_image_eyes', fake_B_eyes)]
            real_B_eyes = render_util.tensor2im(
                render_util.crop_ROI(real_B[0, -1], mc, opt.ROI_size))
            visual_dict += [('real_image_eyes', real_B_eyes)]
        visuals = OrderedDict(visual_dict)
        visualizer.display_current_results(visuals, epoch, iteration)

    return loss_D, loss_G, loss_dict, fake_B_last


def compute_renderer_loss(nets, opt, outs, inputs, device, display=False):
    tgt_mask = inputs['tgt_mask']
    fake_B_last1, fake_B_last2 = None, None
    loss_dict = {}
    loss_D = []
    loss_G = []
    for i, m in enumerate(tgt_mask):
        if m == 0:
            continue
        pseudo = inputs['pseudos'][i]
        src_img_path = inputs['A_paths'][-1][i].replace('nmfcs_aligned', 'images')
        nmfc_list = []
        shape_list = []
        for j in range(opt.n_frames_G):
            # img_path = inputs['A_paths'][j][i].replace(
            #     'nmfcs_aligned', 'images')
            # deca_path = inputs['A_paths'][j][i].replace(
            #     'nmfcs_aligned', 'DECA').replace('png', 'pkl')
            img_path = pseudo['A_paths'][j].replace('nmfcs_aligned', 'images')
            deca_path = pseudo['A_paths'][j].replace('nmfcs_aligned', 'DECA').replace('png', 'pkl')

            with open(deca_path, 'rb') as f:
                codedict = pickle.load(f)
            for k, v in codedict.items():
                if isinstance(v, np.ndarray):
                    codedict[k] = torch.from_numpy(v).to(device)
            nmfc, shape = deca_decoder.decode(outs[i], codedict, img_path)
            nmfc_list.append(torch.from_numpy(nmfc).float())
            shape_list.append(torch.from_numpy(shape).float())

        # generated
        nmfc_video = torch.cat(nmfc_list, dim=-1).permute(2, 0, 1)
        shape_video = torch.cat(shape_list, dim=-1).permute(2, 0, 1)
        rgb_video = pseudo['rgb_video']
        mouth_centers = pseudo['mouth_centers']
        eyes_centers = pseudo['eyes_centers']
        eye_video = pseudo['eye_video']
        mask_video = pseudo['mask_video']
        # mouth_centers = inputs['mouth_centers'][i]
        # eyes_centers = inputs['eyes_centers'][i]
        # eye_video = inputs['eye_video'][i]
        # mask_video = inputs['mask_video'][i]
        loss_D1, loss_G1, loss_dict1, fake_B_last1 = _compute_renderer_loss(
            nets, opt, nmfc_video, rgb_video, mouth_centers, eyes_centers, eye_video, shape_video, mask_video, fake_B_last1, device, display, src_img=src_img_path.replace('images', 'faces_aligned'))
        display = False

        # target
        nmfc_video = pseudo['nmfc_video']
        shape_video = pseudo['shape_video']
        rgb_video = pseudo['rgb_video']
        mouth_centers = pseudo['mouth_centers']
        eyes_centers = pseudo['eyes_centers']
        eye_video = pseudo['eye_video']
        mask_video = pseudo['mask_video']
        loss_D2, loss_G2, loss_dict2, fake_B_last2 = _compute_renderer_loss(
            nets, opt, nmfc_video, rgb_video, mouth_centers, eyes_centers, eye_video, shape_video, mask_video, fake_B_last2, device, display, src_img=src_img_path.replace('images', 'faces_aligned'))
        # display = False

        loss_dict = {
            k: (loss_dict1[k].item()+loss_dict2[k].item())/2
            for k in loss_dict1.keys()
        }
        loss_G += [loss_G1, loss_G2]
        loss_D += [loss_D1, loss_D2]
        # loss_dict = {k: v.item() for k, v in loss_dict2.items()}
        # loss_G += [loss_G2]
        # loss_D += [loss_D2]

    loss_G = torch.stack(loss_G).mean() if len(
        loss_G) else torch.zeros(1, device=device)
    loss_D = torch.stack(loss_D).mean() if len(
        loss_D) else torch.zeros(1, device=device)
    return loss_G, loss_D, Munch(**loss_dict)


if __name__ == '__main__':

    opt = TrainOptions().parse()
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # initialize train dataset
    loader_src = union_dataset.get_train_loader(opt, which='source')
    loader_ref = mead_dataset.get_train_loader(opt, which='reference')

    # initialize val dataset
    loader_src_val = union_dataset.get_val_loader(opt, which='source')

    # initialize models
    nets = create_manipulator_model(opt)
    rendererG, rendererD = create_renderer_model(opt)
    nets.rendererG = rendererG
    nets.rendererD = rendererD

    deca_decoder = DECADecoder(device=device)

    visualizer = Visualizer(opt)

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
        manipulator_weights = torch.load(
            opt.manipulator_pretrain_weight, device)
        nets['generator'].module.load_state_dict(
            manipulator_weights['generator'])
        nets['style_encoder'].module.load_state_dict(
            manipulator_weights['style_encoder'])
        nets['discriminator'].module.load_state_dict(
            manipulator_weights['discriminator'])
        rendererG_weights = torch.load(opt.rendererG_pretrain_weight, device)
        rendererD_weights = torch.load(opt.rendererD_pretrain_weight, device)
        nets['rendererG'].module.netG.load_state_dict(rendererG_weights)
        nets['rendererD'].module.netD.load_state_dict(rendererD_weights)
        if not opt.no_mouth_D:
            rendererDm_weights = torch.load(
                opt.rendererD_pretrain_weight.replace('net_D', 'net_Dm'), device)
            nets['rendererD'].module.netDm.load_state_dict(rendererDm_weights)
        if opt.use_eyes_D:
            rendererDe_weights = torch.load(
                opt.rendererD_pretrain_weight.replace('net_D', 'net_De'), device)
            nets['rendererD'].module.netDe.load_state_dict(rendererDe_weights)
    elif opt.resume:
        load_checkpoint(opt.which_epoch)
    else:
        for name, module in nets.items():
            # mutil.print_network(module, name)
            # print('Initializing %s...' % name)
            module.apply(mutil.he_init)

    loss_log = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
    logfile = open(loss_log, "a")

    fetcher = union_dataset.InputFetcher(loader_src, loader_ref)
    fetcher_val = union_dataset.InputFetcher(loader_src_val, loader_ref)

    start_epoch = opt.which_epoch if opt.resume else 0
    print('Start training...')
    start_time = time.time()
    for epoch in range(start_epoch, opt.niter):
        for model in nets:
            nets[model].train()

        for i in range(len(loader_src)):
            iteration = i + epoch*len(loader_src)

            # fetch sequences and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, y_trg = inputs.x_ref, inputs.y_ref
            x_tgt, tgt_mask = inputs.x_tgt, inputs.tgt_mask

            x_real = x_real.to(device)
            y_org = y_org.to(device)
            x_ref = x_ref.to(device)
            y_trg = y_trg.to(device)
            x_tgt = x_tgt.to(device)
            tgt_mask = tgt_mask.to(device)

            d_loss, d_losses_item = compute_d_loss(
                nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_tgt=x_tgt, tgt_mask=tgt_mask)
            reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_item, x_fake = compute_g_loss(
                nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_tgt=x_tgt, tgt_mask=tgt_mask)
            reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()

            rg_loss, rd_loss, r_losses_item = compute_renderer_loss(
                nets, opt, x_fake, inputs, device, display=(iteration + 1) % opt.display_freq == 0)
            if rg_loss.item() > 0:
                reset_grad()
                rg_loss.backward()
                optims.rendererG.step()
                reset_grad()
                rd_loss.backward()
                optims.rendererD.step()

            # print out log info
            if (i+1) % opt.print_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Epoch [%i/%i], Iteration [%i/%i], " % (
                    elapsed, epoch+1, opt.niter, i+1, len(loader_src))
                all_losses = dict()
                for loss, prefix in zip([d_losses_item, g_losses_item, r_losses_item],  # d_losses_latent, g_losses_latent
                                        ['D/', 'G/', 'Render/']):  # 'D/latent_', 'G/latent_',
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                        logger.log_training(
                            "train/"+prefix+key, value, iteration)
                log += ' '.join(['%s: [%.4f]' % (key, value)
                                for key, value in all_losses.items()])

                print(log)
                logfile.write(log)
                logfile.write('\n')
                logfile.flush()

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
                    x_tgt = inputs.x_tgt
                    tgt_mask = inputs.tgt_mask

                    x_real = x_real.to(device)
                    y_org = y_org.to(device)
                    x_ref = x_ref.to(device)
                    y_trg = y_trg.to(device)
                    x_tgt = x_tgt.to(device)
                    tgt_mask = tgt_mask.to(device)

                    d_loss, d_losses_item = compute_d_loss(
                        nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_tgt=x_tgt, tgt_mask=tgt_mask)

                    g_loss, g_losses_item, x_fake = compute_g_loss(
                        nets, opt, x_real, y_org, y_trg, x_ref=x_ref, x_tgt=x_tgt, tgt_mask=tgt_mask)

                    rg_loss, rd_loss, r_losses_item = compute_renderer_loss(
                        nets, opt, x_fake, inputs, device)

                    # print out log info
                    for loss, prefix in zip([d_losses_item, g_losses_item, r_losses_item],  # d_losses_latent, g_losses_latent
                                            ['D/', 'G/', 'Render/']):  # 'D/latent_', 'G/latent_',
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
