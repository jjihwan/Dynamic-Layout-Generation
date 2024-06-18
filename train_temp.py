import torch
from tqdm import tqdm
from util.datasets.load_data import init_dataset
from util.visualization import save_image
from util.seq_util import sparse_to_dense, pad_until
from model_diffusion import TemporalDiffusion
from util.ema import EMA
import argparse
import pickle as pk
import torch.optim as optim
from util.constraint import *
import math
import os
from test import test_all
from einops import repeat, rearrange
import datetime

def make_dynamic(label, bbox, mask, num_frame=4):
    label = repeat(label, 'b n -> b t n', t=num_frame)
    bbox = repeat(bbox, 'b n d-> b t n d', t=num_frame).clone()
    mask = repeat(mask, 'b n -> b t n', t=num_frame)

    bbox[:, 1, :, 0] = 1 - bbox[:, 0, :, 0]
    bbox[:, 2, :, :2] = 1 - bbox[:, 0, :, :2]
    bbox[:, 3, :, 1] = 1 - bbox[:, 0, :, 1]
    bbox = bbox * mask.unsqueeze(-1)

    return label, bbox, mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=100000, help="number of training epochs", type=int)
    parser.add_argument("--start_epoch", default=0, help="start epoch", type=int)
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--lr", default=1e-5, help="learning rate", type=float)
    parser.add_argument("--sample_t_max", default=999, help="maximum t in training", type=int)
    parser.add_argument("--dataset", default='publaynet',
                        help="choose from [publaynet, rico13, rico25, magazine, crello]", type=str)
    parser.add_argument("--data_dir", default='./datasets', help="dir of datasets", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--n_save_epoch", default=50, help="number of epochs to do test and save model", type=int)
    parser.add_argument("--feature_dim", default=2048, help="feature_dim", type=int)
    parser.add_argument("--dim_transformer", default=1024, help="dim_transformer", type=int)
    parser.add_argument("--embed_type", default='pos', help="embed type for transformer, pos or time", type=str)
    parser.add_argument("--nhead", default=16, help="nhead attention", type=int)
    parser.add_argument("--nlayer", default=4, help="nlayer", type=int)
    parser.add_argument("--align_weight", default=1, help="the weight of alignment constraint", type=float)
    parser.add_argument("--align_type", default='local', help="local or global alignment constraint", type=str)
    parser.add_argument("--overlap_weight", default=1, help="the weight of overlap constraint", type=float)
    parser.add_argument('--load_pre', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--load_pre_spatial', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--beautify', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--enable_test', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gpu_devices", default=[0, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    parser.add_argument("--num_frame", default=4, help="number of frames", type=int)
    parser.add_argument("--wandb", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--project_name", default='LACE-temporal', help="wandb project name", type=str)
    parser.add_argument("--experiment_name", default='publaynet', help="wandb experiment name", type=str)
    args = parser.parse_args()

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f'load_pre: {args.load_pre}, load_pre_spatial: {args.load_pre_spatial}, enable_test: {args.enable_test}, embed: {args.embed_type}')
    print(f'dim_transformer: {args.dim_transformer}, n_layers: {args.nlayer}, nhead: {args.nhead}')
    print(f'align_type: {args.align_type}, align_weight: {args.align_weight}, overlap_weight: {args.overlap_weight}')
    print(f'device: {args.device}, wandb: {args.wandb}, num_frame: {args.num_frame}')

    # prepare data
    if args.embed_type == 'pos':
        train_dataset, train_loader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='train', shuffle=True, transform=None)
    else:
        train_dataset, train_loader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                                   split='train', shuffle=True)

    num_class = train_dataset.num_classes + 1

    # set up model
    model_ddpm = TemporalDiffusion(pretrained_model_path=f"./model/{args.dataset}_best.pt", num_frame=args.num_frame, is_train=True,
                                   num_timesteps=1000, nhead=args.nhead, dim_transformer=args.dim_transformer,
                           feature_dim=args.feature_dim, seq_dim=num_class + 4, num_layers=args.nlayer,
                           device=device, ddim_num_steps=200)

    if args.load_pre:
        # state_dict = torch.load(f'./model/{args.embed_type}_{args.dataset}_1024_recent.pt', map_location='cpu')
        # state_dict = torch.load(f'./model/publaynet_best.pt', map_location='cpu')
        state_dict = torch.load(f'./model/{args.dataset}_temp_best.pt', map_location='cpu')
        model_ddpm.load_diffusion_net(state_dict)
    

    if args.device is None:
        print('using DataParallel')
        model_ddpm.model = nn.DataParallel(model_ddpm.model).to(device)
    else:
        print('using single gpu')
        model_ddpm.to(device)

    if args.load_pre and args.enable_test:
        fid_best = test_all(model_ddpm, dataset_name=args.dataset, seq_dim=num_class + 4, batch_size=args.batch_size,
                            beautify=args.beautify)
        # fid_best = 1e10
    else:
        fid_best = 1e10

    # optimizer
    optimizer = optim.Adam(model_ddpm.model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    mse_loss = nn.MSELoss()

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(model_ddpm.model)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = f'{args.experiment_name}_{now}'
    os.makedirs(f'./model_trained/{nowname}', exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.project_name, name=nowname, config=vars(args))

    for epoch in range(args.start_epoch, args.nepoch):
        model_ddpm.model.train()

        if (epoch) % args.n_save_epoch == 0 and epoch != 0:
            model_path = f'./model_trained/{nowname}/epoch={epoch:06d}.pt'
            states = model_ddpm.model.state_dict()
            torch.save(states, model_path)

            if args.enable_test:
                fid_total = test_all(model_ddpm, dataset_name=args.dataset, seq_dim=num_class + 4, batch_size=args.batch_size, beautify=False)
                # print(f'previous best fid: {fid_best}')
                # if fid_total < fid_best:
                #     # model_path = f'./model/{args.embed_type}_{args.dataset}_1024_lowest.pt'
                #     # torch.save(states, model_path)
                #     fid_best = fid_total
                #     print('New lowest fid model, saved')

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=200) as pbar:

            for i, data in pbar:
                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
                label, bbox, mask = make_dynamic(label, bbox, mask, args.num_frame)
                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                # shift to center
                bbox_in = 2 * (bbox - 0.5).to(args.device)

                # set mask to label 5
                label[mask==False] = num_class - 1

                label_oh = torch.nn.functional.one_hot(label, num_classes=num_class).to(args.device)

                # concat label with bbox and get a 10 dim
                layout_input = torch.cat((label_oh, bbox_in), dim=3).to(args.device)

                t = model_ddpm.sample_t([bbox.shape[0]], t_max=args.sample_t_max)
                t_f = repeat(t, 'b -> (b f)', f=args.num_frame)

                eps_theta, e, b_0_reparam = model_ddpm.forward_t(layout_input, t=t, real_mask=mask, reparam=True) # [B, F, L, D]

                eps_theta = rearrange(eps_theta, 'b f l d -> (b f) l d')
                e = rearrange(e, 'b f l d -> (b f) l d')
                b_0_reparam = rearrange(b_0_reparam, 'b f l d -> (b f) l d')

                # compute b_0 reparameterization
                bbox_rep = torch.clamp(b_0_reparam[:, :, num_class:], min=-1, max=1) / 2 + 0.5

                # gt
                mask = rearrange(mask, 'b f n -> (b f) n', f=args.num_frame)
                bbox = rearrange(bbox, 'b f n d -> (b f) n d', f=args.num_frame)
                # mask_4 = torch.cat([mask, mask, mask, mask], dim=0)
                # bbox_4 = torch.cat([bbox, bbox, bbox, bbox], dim=0)

                # compute alignment loss
                if args.align_type == 'global':
                    # global alignment
                    align_loss = mean_alignment_error(bbox_rep, bbox, mask)
                else:
                    # local alignment
                    _, align_loss = layout_alignment(bbox_rep, mask, xy_only=False)
                    align_loss = 20 * align_loss

                # compute piou and pdist
                piou = PIoU_xywh(bbox_rep, mask=mask.to(torch.float32), xy_only=False)
                pdist = Pdist(bbox_rep)

                # compute piou loss with temporal weight
                overlap_loss = torch.mean(piou, dim=[1, 2]) + torch.mean(piou.ne(0) * torch.exp(-pdist), dim=[1, 2])
                # overlap_loss = torch.mean(piou, dim=[1, 2])

                # reconstruction loss
                # layout_input_all = torch.cat([layout_input, layout_input, layout_input, layout_input], dim=0)
                layout_input = rearrange(layout_input, 'b f l d -> (b f) l d')
                reconstruct_loss = mse_loss(layout_input[:, :, num_class:], b_0_reparam[:, :, num_class:])

                # combine constraints with temporal weight
                weight = constraint_temporal_weight(t_f, schedule='const')

                constraint_loss = torch.mean((args.align_weight * align_loss + args.overlap_weight * overlap_loss)
                                             * weight)

                # compute diffusion loss
                diffusion_loss = mse_loss(e, eps_theta)

                # total loss
                loss = diffusion_loss + constraint_loss + reconstruct_loss

                pbar.set_postfix({'diffusion': diffusion_loss.item(), 'align': torch.mean(align_loss).item(),
                                  'overlap': torch.mean(overlap_loss).item(), 'reconstruct': reconstruct_loss.item()})
                
                if args.wandb:
                    wandb.log({'total': loss.item(), 'diffusion': diffusion_loss.item(), 'align': torch.mean(align_loss).item(),
                               'overlap': torch.mean(overlap_loss).item(), 'reconstruct': reconstruct_loss.item()})

                # optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_ddpm.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(model_ddpm.model)

