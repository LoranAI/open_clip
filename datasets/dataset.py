import torch.utils.data as data
from PIL import Image
import os
import json
import os.path
import pandas as pd
from pycocotools.coco import COCO


class CocoCaptions(data.Dataset):
    """`MS Coco Captions Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annotation_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Example:

        .. code-block::python

            import torchvision.datasets as dataset
            import torchvision.transforms as transforms
            cap = dataset.CocoCaptions(root = 'dir where images are',
                                    annotation_file = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def __init__(self, root, annotation_path, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annotation_path)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']
        print(path)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


def create_csv():
    """
    Note that the Dataset class is not used since the open_clip API wants a single csv file.
    The creation of the csv file is done here.
    """

    root = "/andromeda/datasets/COCO2017/COCO2017_train/train2017"
    path = "/andromeda/datasets/COCO2017/COCO2017_train/annotations/captions_train2017.json"

    coco = COCO(path)
    ids = list(coco.imgs.keys())
    imgs = coco.loadImgs(coco.getImgIds())
    future_df = {"filepath": [], "title": []}
    for img, id in zip(imgs, ids):
        ann_ids = coco.getAnnIds(imgIds=id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            future_df["filepath"].append(img["file_name"])
            future_df["title"].append(ann["caption"])

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(root, "train2017.csv"), index=False, sep="\t"
    )


def create_json(root_path, path_root, root_project, new_file="train2017.json"):
    coco = COCO(path_root)

    ids = list(coco.imgs.keys())
    imgs = coco.loadImgs(ids)

    data = []

    for img in imgs:
        element = {}
        img_id = img['id']
        file_name = img['file_name']

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        captions = [ann['caption'] for ann in anns]

        element["image"] = os.path.join(root_path, file_name)
        element["caption"] = captions

        data.append(element)

    output_path = os.path.join(root_project, new_file)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    root_project = "./datasets"

    root_train = "/andromeda/datasets/COCO2017/COCO2017_train/train2017"
    path_train = "/andromeda/datasets/COCO2017/COCO2017_train/annotations/captions_train2017.json"

    root_val = "/andromeda/datasets/COCO2017/COCO2017_val/val2017"
    path_val = "/andromeda/datasets/COCO2017/COCO2017_val/annotations/captions_val2017.json"

    new_file_val = "val2017.json"
    new_file_train = "train2017.json"

    create_json(root_train, path_train, root_project, new_file_train)
    create_json(root_val, path_val, root_project, new_file_val)
