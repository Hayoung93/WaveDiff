import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class DehazeCityscapesDataset(Dataset):
    def __init__(self, root="/data/data", mode="train", transform=None):
        assert mode in ["train", "val", "test"]
        self.root = root
        self.mode = mode
        self.transform = transform

        # clear_json = os.path.join(root, "cityscapes/annotations/instancesonly_filtered_gtFine_{}_poly.json".format(mode))
        # haze_json = os.path.join(root, "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_{}_poly.json".format(mode))

        self.clear_image_dir = os.path.join(root, "cityscapes/leftImg8bit/{}".format(mode))
        self.haze_image_dir = os.path.join(root, "foggy_cityscapes/leftImg8bit_foggy/{}".format(mode))

        self.images_haze = []

        for (rootdir, dirs, files) in os.walk(self.haze_image_dir):
            for file in files:
                self.images_haze.append(os.path.join(self.haze_image_dir, rootdir, file))
    
    def __len__(self):
        return len(self.images_haze)
    
    def __getitem__(self, idx):
        img_haze_fp = self.images_haze[idx]
        img_haze_fn = img_haze_fp.split("/")[-1]
        img_haze_fn_split = img_haze_fn.split("_")
        img_clear_fp = os.path.join(self.clear_image_dir, img_haze_fn_split[0], "_".join(img_haze_fn_split[:4]) + ".png")
        assert os.path.exists(img_clear_fp), "No clear image founded"

        img_clear = Image.open(img_clear_fp).convert("RGB")
        img_haze = Image.open(img_haze_fp).convert("RGB")

        if self.transform is not None:
            img_clear = self.transform(img_clear)
            img_haze = self.transform(img_haze)
        
        return img_haze, img_clear


class RSHaze(Dataset):
    def __init__(self, root="/data/data/RSHaze", mode="train", transform=None):
        assert mode in ["train", "test", "test_small"]
        self.root = root
        self.mode = mode
        self.transform = transform

        self.hazy_dir = os.path.join(root, mode, "hazy")
        self.image_list_hazy = os.listdir(self.hazy_dir)
        self.gt_dir = os.path.join(root, mode, "GT")
        assert os.path.exists(self.gt_dir)

    def __getitem__(self, idx):
        haze_fn = self.image_list_hazy[idx]
        haze_fp = os.path.join(self.hazy_dir, haze_fn)
        img_haze = Image.open(haze_fp).convert("RGB")
        gt_fp = os.path.join(self.gt_dir, haze_fn)
        img_gt = Image.open(gt_fp).convert("RGB")

        if self.transform is not None:
            img_haze, img_gt = self.transform(img_haze, img_gt)
        
        return img_haze, img_gt

    def __len__(self):
        return len(self.image_list_hazy)


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
