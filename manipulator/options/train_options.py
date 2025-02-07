from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume training')
        self.parser.add_argument('--finetune', action='store_true', help='Whether to perform training from scratch or finetuning')
        self.parser.add_argument('--finetune_epoch', type=int, default=20, help='Epoch to load checkpoint for finetuning')

        self.parser.add_argument('--niter', type=int, default=10, help='# of total epochs')
        self.parser.add_argument('--niter_decay', type=int, default=10, help='# of epochs for decaying lr')
        self.parser.add_argument('--val_freq', type=int, default=5, help='# val frequence')

        self.parser.add_argument('--beta1', type=float, default=0., help='decay rate for 1st moment of Adam')
        self.parser.add_argument('--beta2', type=float, default=0.99, help='decay rate for 2nd moment of Adam')
        self.parser.add_argument('--g_lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--d_lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--e_lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--f_lr', type=float, default=1e-4, help='initial learning rate for mapping network')

        self.parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cycle consistency loss')
        self.parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
        self.parser.add_argument('--lambda_mouth', type=float, default=1, help='Weight for mouth loss')
        self.parser.add_argument('--lambda_paired', type=float, default=1, help='Weight for paired loss')

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

        self.parser.add_argument('--paired', action='store_true', help='Whether to do paired training')
        
        self.isTrain = True
