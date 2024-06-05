import numpy as np
from tensorboardX import SummaryWriter
import logging
import argparse
import torch
from datetime import datetime
from torch.utils import data
from floortrans.models import get_model
from tqdm import tqdm
import glob
import cv2
from floortrans import post_prosessing
from matplotlib import pyplot as plt
import os

room_cls = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
icon_cls = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]


class ImageDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = glob.glob(data_dir + '/*.png')
        self.image_files += glob.glob(data_dir + '/*.jpg')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        # transform to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image

    def transform(self, image):
        # Normalization values to range -1 and 1
        image = 2 * (image / 255.0) - 1
        return image


def visualize_prediction(image, rooms, icons, output_file):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Overlay room predictions
    for i, room in enumerate(room_cls):
        if room != "Wall":
            continue
        room_mask = rooms[i, :, :] > 0.5
        ax.imshow(np.ma.masked_where(~room_mask, room_mask), cmap='jet', alpha=0.5)

    # Overlay icon predictions (Window and Door)
    for i, icon in enumerate(icon_cls):
        if icon not in ["Window", "Door"]:
            continue
        icon_mask = icons[i, :, :] > 0.5
        ax.imshow(np.ma.masked_where(~icon_mask, icon_mask), cmap='hot', alpha=0.5)

    plt.axis('off')
    plt.savefig(output_file)
    plt.close()


def evaluate(args, log_dir, writer, logger):

    normal_set = ImageDataset(args.data_dir)
    data_loader = data.DataLoader(normal_set, batch_size=1, num_workers=0)

    checkpoint = torch.load(args.weights)
    # Setup Model
    model = get_model(args.arch, 51)
    n_classes = args.n_classes
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()

    # First 12 class of room, last 11 class of icon
    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            val = val.cuda()
            logger.info(count)
            pred = model(val)
            h, w = val.size(2), val.size(3)
            pred = pred.detach().cpu()
            heatmaps, rooms, icons = post_prosessing.split_prediction(
                pred, (h, w), split)
            image = val.cpu().numpy()[0].transpose(1, 2, 0)
            image = ((image + 1) / 2 * 255).astype(np.uint8)  # Denormalize to original image
            # rooms: 12, icons: 11
            
            output_file = os.path.join(log_dir, f"visualization_{count}.png")
            visualize_prediction(image, rooms, icons, output_file)



if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Settings for evaluation')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet etc\']')
    parser.add_argument('--data-dir', nargs='?', type=str, default='custom_data',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')

    args = parser.parse_args()

    log_dir = args.log_path + '/' + time_stamp + '/'
    writer = SummaryWriter(log_dir)
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/eval.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    evaluate(args, log_dir, writer, logger)
