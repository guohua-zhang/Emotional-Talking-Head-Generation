import argparse
import os
import torch

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--seq_len', type=int, default=10, help='Length of exp. coeffs. sequence')
        self.parser.add_argument('--hop_len', type=int, default=1, help='Hop Length (set to 1 by default for test)')
        self.parser.add_argument('--selected_emotions', type=str, nargs='+', help='Subset (or all) of the 8 basic emotions',
                                 default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'],
                                 choices=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt'])
        self.parser.add_argument('--train_root', type=str, default='./MEAD_data', help='Directory containing (reconstructed) MEAD')
        self.parser.add_argument('--selected_actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003','M009','W029'])
        self.parser.add_argument('--selected_actors_val', type=str, nargs='+', help='Subset of the MEAD actors', default=['M023'])
        self.parser.add_argument('--selected_actors_ref', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003','M009','W029'])
        self.parser.add_argument('--selected_emotions_ref', type=str, nargs='+', help='Subset (or all) of the 8 basic emotions',
                                 default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'],
                                 choices=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt'])
        self.parser.add_argument('--class_names', type=str, nargs='+', help='Subset (or all) of the 8 basic emotions',
                                 default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'],
                                 choices=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt'])
        self.parser.add_argument('--ref_dirs', type=str, nargs='+', help='Directories containing input reference sequences', default=None)
        self.parser.add_argument('--test_dir', type=str, help='Directories containing input test sequences, for training visualization', default=None)
        self.parser.add_argument('--dist_file', type=str, help='audio distances for training dataset', default=None)
        self.parser.add_argument('--dist_thresh', type=float, help='audio distances threshold', default=1.0)

        self.parser.add_argument('--paired', type=bool, default=True, help='Whether to do paired training')

        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of mapping network')
        self.parser.add_argument('--style_dim', type=int, default=16, help='Style code dimension')

        self.parser.add_argument('--no_align', action='store_true', help='if specified, use original face images (not aligned)')
        self.parser.add_argument('--max_n_sequences', type=int, default=None, help='Maximum number of sub-sequences to use.')
        self.parser.add_argument('--no_augment_input', action='store_true', help='if true, do not perform input data augmentation.')
        self.parser.add_argument('--ROI_size', type=int, default=72, help='spatial dimension size of ROI (mouth or eyes).')
        self.parser.add_argument('--no_mouth_D', action='store_true', help='if true, do not use mouth discriminator')
        self.parser.add_argument('--use_eyes_D', action='store_true', help='if true, Use eyes discriminator')
        self.parser.add_argument('--no_eye_gaze', action='store_true', help='if true, the model does not condition synthesis on eye gaze images')
        self.parser.add_argument('--use_shapes', type=bool, default=True, help='if True, the model conditions synthesis on shape images as well')
        self.parser.add_argument('--n_frames_G', type=int, default=3, help='number of input frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')
        self.parser.add_argument('--n_frames_total', type=int, default=1, help='the overall number of frames in a sequence to train with')
        self.parser.add_argument('--max_frames_per_gpu', type=int, default=4, help='max number of frames to load into one GPU at a time')
        self.parser.add_argument('--n_frames_backpropagate', type=int, default=1, help='max number of frames to backpropagate')
        self.parser.add_argument('--no_first_img', action='store_true', default=True, help='if specified, generator synthesizes the first image')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='if specified, load the pretrained model')
        self.parser.add_argument('--resize', action='store_true', default=True, help='Resize the input images to loadSize')

        self.parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--input_nc', type=int, default=9, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--val_freq', type=int, default=5, help='# val frequence')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--resume', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--finetune', type=bool, default=True, help='continue training: load the latest model')
        self.parser.add_argument('--manipulator_pretrain_weight', type=str, default='', help='manipulator pretrain weight')
        self.parser.add_argument('--rendererG_pretrain_weight', type=str, default='', help='rendererG pretrain weight')
        self.parser.add_argument('--rendererD_pretrain_weight', type=str, default='', help='rendererD pretrain weight')
        self.parser.add_argument('--which_epoch', type=int, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=10, help='# of epochs at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of epochs to linearly decay learning rate to zero')

        self.parser.add_argument('--beta1', type=float, default=0., help='decay rate for 1st moment of Adam')
        self.parser.add_argument('--beta2', type=float, default=0.99, help='decay rate for 2nd moment of Adam')
        self.parser.add_argument('--g_lr', type=float, default=1e-5, help='initial learning rate for G network')
        self.parser.add_argument('--d_lr', type=float, default=1e-5, help='initial learning rate for D network')
        self.parser.add_argument('--e_lr', type=float, default=1e-5, help='initial learning rate for style_encoder')
        self.parser.add_argument('--rg_lr', type=float, default=1e-5, help='initial learning rate for renderer G network')
        self.parser.add_argument('--rd_lr', type=float, default=1e-5, help='initial learning rate for renderer D network')

        self.parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cycle consistency loss')
        self.parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
        self.parser.add_argument('--lambda_mouth', type=float, default=1, help='Weight for mouth loss')
        self.parser.add_argument('--lambda_paired', type=float, default=1, help='Weight for paired loss')
        self.parser.add_argument('--lambda_paired_renderer', type=float, default=1, help='Weight for paired loss')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for vgg and feature matching')
        
        self.parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|other), with other being a hinge loss')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/web/')
        self.parser.add_argument('--num_D', type=int, default=2, help='number of patch scales in each discriminator')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')
        self.parser.add_argument('--no_vgg', action='store_true', help='do not use VGG feature matching loss')
        self.parser.add_argument('--no_ganFeat', action='store_true', help='do not match discriminator features')
        self.parser.add_argument('--no_prev_output', action='store_true', help='if true, do not use the previously generated frames in G input.')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--n_blocks', type=int, default=9, help='number of resnet blocks in generator')
        self.parser.add_argument('--n_downsample_G', type=int, default=3, help='number of downsampling layers in netG')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./Pacino/checkpoints', help='models are saved here')
        self.parser.add_argument('--exp_name', type=str, default='', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, perform shuffling in path creation, otherwise in the dataloader. Set in case of frequent out of memory exceptions.')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.parser.add_argument('--seed', type=int, default=2023, help='set random seed')
        self.initialized = True
        self.isTrain = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        os.makedirs(self.opt.checkpoints_dir, exist_ok=True)
        if save:
            file_name = os.path.join(self.opt.checkpoints_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
