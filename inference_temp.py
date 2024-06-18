import os
# from fid.model import load_fidnet_v3
from util.metric import compute_generative_model_scores, compute_maximum_iou, compute_overlap, compute_alignment
import pickle as pk
from tqdm import tqdm
from util.datasets.load_data import init_dataset
from util.visualization import save_image
from util.constraint import *
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from util.seq_util import sparse_to_dense, loader_to_list, pad_until
import argparse
from model_diffusion import Diffusion, TemporalDiffusion
import imageio
from train_temp import make_dynamic


# def test_fid_feat(dataset_name, device='cuda', batch_size=20):

#     if os.path.exists(f'./fid/feature/fid_feat_test_{dataset_name}.pk'):
#         feats_test = pk.load(open(f'./fid/feature/fid_feat_test_{dataset_name}.pk', 'rb'))
#         return feats_test

#     # prepare dataset
#     main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
#                                                  split='test', shuffle=False, transform=None)

#     fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
#     feats_test = []

#     with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'Get feature for FID',
#               ncols=200) as pbar:

#         for i, data in pbar:

#             bbox, label, _, mask = sparse_to_dense(data)
#             label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)
#             padding_mask = ~mask

#             with torch.set_grad_enabled(False):
#                 feat = fid_model.extract_features(bbox, label, padding_mask)
#             feats_test.append(feat.detach().cpu())

#     pk.dump(feats_test, open(f'./fid/feature/fid_feat_test_{dataset_name}.pk', 'wb'))

#     return feats_test




def test_layout_cond(model, batch_size=256, cond='c', dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test', beautify=False):
    """
    Test the model with conditional generation
    :param model: the model to test
    :param batch_size: batch size
    :param cond: condition type, choose from ['c', 'cwh', 'complete']
    :param dataset_name: choose from ['publaynet', 'rico13', 'rico25']
    :param seq_dim: sequence dimension, N+5
    :param test_plot: whether to plot the generated layout
    :param save_dir: directory to save the plot
    :param beautify: whether to beautify the layout
    :return: align_final, fid, maxiou, overlap_final
    """

    assert cond in {'c', 'cwh', 'complete'}
    model.eval()
    device = model.device

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    layout_generated = []

    # fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    # feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0

    if test_plot:
        os.makedirs(os.path.join(save_dir, f"{dataset_name}_temp"), exist_ok=True)

    with torch.no_grad():
        # with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'cond: {cond} generation',
        #           ncols=200) as pbar:
        with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'cond: {cond} generation',
                  ncols=200) as pbar:

            for i, data in pbar:
                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
                label, bbox, mask = make_dynamic(label, bbox, mask, args.num_frame)
                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                # shift to center
                bbox_in = 2 * (bbox - 0.5).to(device)

                # set mask to label N
                label[mask == False] = seq_dim - 5

                label_oh = torch.nn.functional.one_hot(label, num_classes=seq_dim - 4).to(device)
                real_layout = torch.cat((label_oh, bbox_in), dim=3).to(device)

                bbox_generated, label_generated, mask_generated = model.conditional_reverse_ddim(real_layout, cond=cond)

                # if beautify and dataset_name == 'publaynet':
                #     bbox_generated, mask_generated = post_process(bbox_generated, mask_generated, w_o=1)
                # elif beautify and (dataset_name == 'rico25' or dataset_name == 'rico13'):
                #     bbox_generated, mask_generated = post_process(bbox_generated, mask_generated, w_o=0)

                # padding_mask = ~mask_generated

                # test for errors
                if torch.isnan(bbox[0, 0, 0]):
                    print('not a number error')
                    return None

                # accumulate align and overlap
                align_norm = compute_alignment(bbox_generated, mask)
                align_sum += torch.mean(align_norm)
                overlap_score = compute_overlap(bbox_generated, mask)
                overlap_sum += torch.mean(overlap_score)

                # record for max_iou
                # label_generated[label_generated == seq_dim - 5] = 0
                # for j in range(bbox.shape[0]):
                #     mask_single = mask_generated[j, :]
                #     bbox_single = bbox_generated[j, mask_single, :]
                #     label_single = label_generated[j, mask_single]

                #     layout_generated.append((bbox_single.to('cpu').numpy(), label_single.to('cpu').numpy()))

                # record for FID
                # with torch.set_grad_enabled(False):
                #     feat = fid_model.extract_features(bbox_generated, label_generated, padding_mask)
                # feats_generate.append(feat.cpu())

                if test_plot and i <= 10:
                    imgs_generated = []
                    imgs_gt = []
                    for j in range(bbox_generated.shape[1]):
                        img = save_image(bbox_generated[:9,j], label_generated[:9,j], mask_generated[:9,j],
                                        draw_label=False, dataset=dataset_name)
                        imgs_generated.append(img)
                        plt.figure(figsize=[12, 12])
                        plt.imshow(img)
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'{dataset_name}_temp/{i}_{j}.png'))
                        plt.close()

                        img = save_image(bbox[:9,j], label[:9,j], mask[:9,j], draw_label=False, dataset=dataset_name)
                        imgs_gt.append(img)
                        plt.figure(figsize=[12, 12])
                        plt.imshow(img)
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'{dataset_name}_temp/{i}_{j}_gt.png'))
                        plt.close()
                    
                    # make gif
                    imageio.mimsave(os.path.join(save_dir, f'{dataset_name}_temp/{i}.gif'), imgs_generated, loop=0)
                    imageio.mimsave(os.path.join(save_dir, f'{dataset_name}_temp/{i}_gt.gif'), imgs_gt, loop=0)

    # maxiou = compute_maximum_iou(layouts_main, layout_generated)
    # result = compute_generative_model_scores(feats_test, feats_generate)
    # fid = result['fid']

    # align_final = 100 * align_sum / len(main_dataloader)
    # overlap_final = 100 * overlap_sum / len(main_dataloader)

    # print(f'cond {cond}, align: {align_final}, fid: {fid}, maxiou: {maxiou}, overlap: {overlap_final}')

    # return align_final, fid, maxiou, overlap_final
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--dataset", default='publaynet',
                        help="choose from [publaynet, rico13, rico25]", type=str)
    parser.add_argument("--pretrained_model_path", default='model_trained/publaynet_2024-06-18T11-03-00/epoch=000020.pt', type=str)
    parser.add_argument("--data_dir", default='./datasets', help="dir of datasets", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--feature_dim", default=2048, help="feature_dim", type=int)
    parser.add_argument("--dim_transformer", default=1024, help="dim_transformer", type=int)
    parser.add_argument("--nhead", default=16, help="nhead attention", type=int)
    parser.add_argument("--nlayer", default=4, help="nlayer", type=int)
    parser.add_argument("--experiment", default='c', help="experiment setting [uncond, c, cwh, complete, all]", type=str)
    parser.add_argument('--plot', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--beautify', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_save_dir", default='./plot', help="dir to save generated plot of layouts", type=str)
    parser.add_argument("--num_frame", default=4, help="number of frames for diffusion", type=int)

    args = parser.parse_args()

    os.makedirs(args.plot_save_dir, exist_ok=True)

    # prepare data
    train_dataset, train_loader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='train', shuffle=True)
    num_class = train_dataset.num_classes + 1 # including padding; N+1

    # set up model
    model_ddpm = TemporalDiffusion(pretrained_model_path=args.pretrain_model_path, num_frame=args.num_frame, is_train=False,
                            num_timesteps=1000, nhead=args.nhead, dim_transformer=args.dim_transformer,
                           feature_dim=args.feature_dim, seq_dim=num_class + 4, num_layers=args.nlayer,
                           device=args.device, ddim_num_steps=100)

    # state_dict = torch.load(f'./model/{args.dataset}_best.pt', map_location='cpu')
    # model_ddpm.load_diffusion_net(state_dict)


    test_layout_cond(model_ddpm, batch_size=args.batch_size, cond=args.experiment,
                                        dataset_name=args.dataset, seq_dim=num_class + 4,
                                        test_plot=args.plot, save_dir=args.plot_save_dir, beautify=args.beautify)




