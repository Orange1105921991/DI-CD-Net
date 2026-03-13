# -*- coding: windows-1252 -*-
from argparse import ArgumentParser
# import torch
from models.trainer import *
import os
import datetime
import random
print(torch.cuda.is_available())

def set_seed(seed):
    random.seed(seed)                      # ÉèÖÃ Python ÄÚÖÃµÄËæ»úÊýÉú³ÉÆ÷µÄÖÖ×Ó
    np.random.seed(seed)                   # ÉèÖÃ NumPy µÄËæ»úÊýÉú³ÉÆ÷µÄÖÖ×Ó
    torch.manual_seed(seed)                # ÉèÖÃ PyTorch µÄ CPU Ëæ»úÊýÖÖ×Ó
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       # ÉèÖÃ PyTorch GPU µÄËæ»úÊýÖÖ×Ó
        torch.cuda.manual_seed_all(seed)   # Èç¹ûÓÐ¶à¸ö GPU£¬ÉèÖÃËùÓÐ GPU µÄÖÖ×Ó
    torch.backends.cudnn.deterministic = True   # È·±£Ã¿´Î½á¹ûÒ»ÖÂ
    torch.backends.cudnn.benchmark = False      # ½ûÓÃ CuDNN µÄ×Ô¶¯µ÷ÓÅ

def train(args):
    set_seed(args.seed)  # ÔÚÑµÁ·¿ªÊ¼Ç°ÉèÖÃËæ»úÖÖ×Ó        
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()
def test(args):
    set_seed(args.seed)  # ÔÚÑµÁ·¿ªÊ¼Ç°ÉèÖÃËæ»úÖÖ×Ó    
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------1
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='elgcnet_levir', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='./vis', type=str)
    parser.add_argument('--seed', default=11, type=int)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--dec_embed_dim', default=256, type=int)
    parser.add_argument('--pretrain', default=None, type=str)

    parser.add_argument('--net_G', default='DI-CD-segnet', type=str,
                        help='DI-CD-segnet')
    parser.add_argument('--N', default='5', type=str,
                        help='1-5')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.00031, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=[100], type=int)
    
    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')

    save_dir = f'{time_str}_bs{args.batch_size}_{args.net_G}_N{args.N}_{args.optimizer}_{args.lr_policy}_nums2_442'

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(save_dir, args.project_name) 
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
