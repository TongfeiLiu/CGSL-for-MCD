import numpy as np
import argparse
import torch
import torch.nn.functional as F
from skimage.filters.thresholding import threshold_otsu
from skimage import io
import os
from skimage.segmentation import slic, mark_boundaries
from lalacin_kerneal import laplacian_kernel
from evaluation import Evaluation
from preprocess import process_images
from normal_edge import feature_normalization
from model import Encoder, Decoder_a, Decoder_b
from loss_py import GAE_Loss
# Argument parser
parser = argparse.ArgumentParser(description='CGSL Change Detection')
parser.add_argument('--root_dir', type=str, default='', help='root directory')
parser.add_argument('--load_data_dir', type=str, default='', help='directory to load dataset')
parser.add_argument('--result_folder', type=str, default='', help='folder to save results')
parser.add_argument('--img_t1_name', type=str, default='', help='T1 image filename')
parser.add_argument('--img_t2_name', type=str, default='', help='T2 image filename')
parser.add_argument('--ref_name', type=str, default='', help='reference/ground truth image filename')
parser.add_argument('--type_a', type=str, default='', help='type of first image for preprocessing')
parser.add_argument('--type_b', type=str, default='', help='type of second image for preprocessing')
parser.add_argument('--n_seg', type=int, default=0, help='number of segments for SLIC')
parser.add_argument('--com', type=float, default=0, help='compactness for SLIC')
parser.add_argument('--item', type=int, default=5, help='item index')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--lambda_reg', type=float, default=0.001, help='lambda regularization')
args = parser.parse_args()

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_images(data_path, img_t1_name, img_t2_name, ref_name):
    """
    Load T1, T2 and reference images.
    - Keep T1 and T2 with original channel count (SAR may be single-channel, OPT may be 3-channel).
    - Ensure ref is single-channel binary (0, 255).
    """
    a = io.imread(os.path.join(data_path, img_t1_name))
    b = io.imread(os.path.join(data_path, img_t2_name))
    ref = io.imread(os.path.join(data_path, ref_name))

    # Ensure reference is single-channel
    if ref.ndim == 3:
        ref = ref[..., 0]
    ref = ref.astype(np.uint8)
    ref[ref < 200] = 0
    ref[ref >= 200] = 255

    # Ensure T1/T2 have shape (H, W, C)
    if a.ndim == 2:
        a = a[:, :, np.newaxis]  # (H, W, 1)
    if b.ndim == 2:
        b = b[:, :, np.newaxis]

    # Convert to torch float32
    a = torch.from_numpy(a).to(device)
    b = torch.from_numpy(b).to(device)

    return a, b, ref


def main():
    load_data = os.path.join(args.root_dir + args.load_data_dir)
    img_t1, img_t2, ref = load_images(load_data, args.img_t1_name, args.img_t2_name, args.ref_name)
    height, width, in_channels = img_t1.shape

    # --- Superpixel segmentation (SLIC) ---
    objects = slic(img_t2.cpu().numpy(), n_segments=args.n_seg, compactness=args.com, start_label=0, channel_axis=-1)
    objects = torch.from_numpy(objects).to(device)

    # --- Visualize SLIC segmentation boundaries ---
    img_t2_np = img_t2.cpu().numpy()
    if img_t2_np.shape[-1] == 1:  # single-channel
        img_t2_np = np.repeat(img_t2_np, 3, axis=-1)  # (H, W, 3)
    di_sp_b = mark_boundaries(img_t2_np, objects.cpu().numpy())

    file_path = os.path.join(args.result_folder, f'di_sp_{args.n_seg}_{args.com}_{args.item}')
    os.makedirs(file_path, exist_ok=True)

    io.imsave(
        os.path.join(file_path, f'di_sp_slic{args.n_seg}_{args.com:.1f}.bmp'),
        (di_sp_b * 255).astype(np.uint8)
    )

    # --- Preprocessing ---
    img_t1, img_t2 = process_images(img_t1, img_t2, type_a=args.type_a, type_b=args.type_b)

    # --- Build node sets and adjacency matrices ---
    obj_nums = torch.max(objects) + 1
    node_set_t1, node_set_t2, am_set_t1, am_set_t2 = [], [], [], []

    for idx in range(obj_nums):
        obj_idx = objects == idx
        patch_t1 = img_t1[obj_idx]
        patch_t2 = img_t2[obj_idx]

        if patch_t1.ndim == 1:
            patch_t1 = patch_t1[:, np.newaxis]
            patch_t2 = patch_t2[:, np.newaxis]

        node_set_t1.append(patch_t1)
        node_set_t2.append(patch_t2)

        edge_a = laplacian_kernel(patch_t1)
        edge_b = laplacian_kernel(patch_t2)

        norm_adj_t1 = feature_normalization(edge_a)
        norm_adj_t2 = feature_normalization(edge_b)
        am_set_t1.append(norm_adj_t1)
        am_set_t2.append(norm_adj_t2)

    # --- Encoder/Decoder with adaptive input channels ---
    encoder = Encoder(in_channels=in_channels, hidden_channels=3, out_channels=2).to(device)
    decoder_a = Decoder_a(in_channels=2, hidden_channels=3, out_channels=in_channels).to(device)
    decoder_b = Decoder_b(in_channels=2, hidden_channels=3, out_channels=in_channels).to(device)
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d_a = torch.optim.Adam(decoder_a.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d_b = torch.optim.Adam(decoder_b.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = GAE_Loss(lambda_reg=args.lambda_reg)

    # --- Training function ---
    def train(epoch):
        encoder.train()
        decoder_a.train()
        decoder_b.train()
        optimizer_e.zero_grad()
        optimizer_d_a.zero_grad()
        optimizer_d_b.zero_grad()
        total_loss = 0

        for _iter in range(obj_nums):
            node_t1 = node_set_t1[_iter].to(device)
            norm_adj_t1 = am_set_t1[_iter].to(device)
            node_t2 = node_set_t2[_iter].to(device)
            norm_adj_t2 = am_set_t2[_iter].to(device)

            x_a, mean_a, logstd_a, _, _, _ = encoder(node_t1, norm_adj_t1)
            z = mean_a + torch.exp(logstd_a) * torch.randn_like(logstd_a)
            adj_a = decoder_a(z, norm_adj_t1)

            x_b, mean_b, logstd_b, _, _, _ = encoder(node_t2, norm_adj_t2)
            w = mean_b + torch.exp(logstd_b) * torch.randn_like(logstd_b)
            adj_b = decoder_b(w, norm_adj_t2)

            loss = criterion(node_t1, adj_a, node_t2, adj_b, mean_a, logstd_a, mean_b, logstd_b,
                             encoder, x_a, x_b, decoder_a, decoder_b)
            loss.backward()
            optimizer_e.step()
            optimizer_d_a.step()
            optimizer_d_b.step()
            total_loss += loss.item()

        avg_loss = total_loss / obj_nums
        weight_path = os.path.join(file_path, f'weight_epoch_{epoch}.pth')
        model_dict = {'encoder': encoder.state_dict(),
                      'decoder_a': decoder_a.state_dict(),
                      'decoder_b': decoder_b.state_dict()}
        torch.save(model_dict, weight_path)
        print(f'Epoch [{epoch}], Loss: {avg_loss}')

    # --- Testing function ---
    def test(epoch):
        encoder.eval()
        decoder_a.eval()
        decoder_b.eval()
        with torch.no_grad():
            diff_set_2 = []
            for _iter in range(obj_nums):
                node_t1 = node_set_t1[_iter].to(device)
                norm_adj_t1 = am_set_t1[_iter].to(device)
                node_t2 = node_set_t2[_iter].to(device)
                norm_adj_t2 = am_set_t2[_iter].to(device)

                x_a, mean_a, logstd_a, _, _, _ = encoder(node_t1, norm_adj_t1)
                x_b, mean_b, logstd_b, _, _, _ = encoder(node_t2, norm_adj_t2)

                diff_set_2.append(F.mse_loss(mean_a, mean_b))

        # --- Generate difference map ---
        if in_channels == 1:
            diff_map = torch.zeros((height, width), device=device)
        else:
            diff_map = torch.zeros((height, width, in_channels), device=device)

        for i in range(obj_nums):
            diff_map[objects == i] = diff_set_2[i]

        diff_map_nor = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-10)

        # --- Save difference map ---
        diff_map_save = (diff_map_nor * 255).cpu().numpy().astype(np.uint8)
        io.imsave(os.path.join(file_path, f"DI_iter_{epoch}.png"), diff_map_save)

        # --- Threshold to get change map ---
        CM1 = ((diff_map_save >= threshold_otsu(diff_map_save)) * 255).astype(np.uint8)
        if in_channels > 1:
            CM1 = CM1[..., 0]  # evaluation requires single-channel
        io.imsave(os.path.join(file_path, f"CM_iter_{epoch}.png"), CM1)

        # --- Evaluation ---
        Indicators1 = Evaluation(ref, CM1)
        OA1, kappa1, AA1 = Indicators1.Classification_indicators()
        P1, R1, F11 = Indicators1.ObjectExtract_indicators()
        TP1, TN1, FP1, FN1 = Indicators1.matrix()

        with open(os.path.join(file_path, "eval.txt"), 'a') as val_acc:
            val_acc.write('=============================== Parameters settings ==============================\n')
            val_acc.write(f'=== epoch={epoch} || superpixel Num={args.n_seg} || compact={args.com} ===\n')
            val_acc.write('Domain t1:\n')
            val_acc.write(f'TP={TP1} || TN={TN1} || FP={FP1} || FN={FN1}\n')
            val_acc.write(f"\"OA\":\"{OA1}\"\n")
            val_acc.write(f"\"Kappa\":\"{kappa1}\"\n")
            val_acc.write(f"\"AA\":\"{AA1}\"\n")
            val_acc.write(f"\"Precision\":\"{P1}\"\n")
            val_acc.write(f"\"Recall\":\"{R1}\"\n")
            val_acc.write(f"\"F1\":\"{F11}\"\n")

    for epoch in range(args.epochs):
        train(epoch)  # uncomment to train
        test(epoch)


if __name__ == '__main__':
    main()
