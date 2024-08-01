import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from open_clip_train.distributed import is_master
from open_clip_train.train import AverageMeter, get_retrieval_scores_batched 

import wandb


# TODO implement the new version of retrival scores
def get_clip_too_metrics():
    if isinstance(scores, tuple):
        scores_i2t = scores[0]
        scores_t2i = scores[1].T  # Make it N_ims x N_text

    else:
        scores_t2i = scores
        scores_i2t = scores

    print(f"COCO results across {scores_i2t.shape} samples. ")
    prec_at_1 = AverageMeter()
    prec_at_5 = AverageMeter()

    # Text retrieval
    tqdm_iterator = tqdm(range(len(self.img2txt)))
    for i in tqdm_iterator:
        top5_captions = np.argsort(scores_i2t[i])[-5:]
        true_captions = self.img2txt[i]

        prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:])) > 0)
        prec_at_5.update(len(set(true_captions) & set(top5_captions)) > 0)

        tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f}")

    # Image Retrieval
    image_prec_at_1 = AverageMeter()
    image_prec_at_5 = AverageMeter()

    tqdm_iterator = tqdm(range(len(self.txt2img)))
    for i in tqdm_iterator:
        top5_images = np.argsort(scores_t2i[:, i])[-5:]
        true_image = self.txt2img[i]

        image_prec_at_1.update(true_image in top5_images[-1:])
        image_prec_at_5.update(true_image in top5_images)

        tqdm_iterator.set_description(
            f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f}")

    records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg, "TextPrec@1": prec_at_1.avg,
                "TextPrec@5": prec_at_5.avg}]
    return records


def evaluate_COCO2017(model, dataset, epoch, args):
    dataset = dataset['coco2017']
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    scores = model.get_retrieval_scores_dataset(loader)
    result_records = dataset.evaluate_scores(scores)
    # Log the results on wandb
    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        res = result_records[0]
        log_data = {"recall/" + name: val for name, val in res.items()}
        log_data['epoch'] = epoch
        wandb.log(log_data)


def evaluate_ARO(model, data, tokenizer, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    for dataset_name, dataset in data["aro_eval"].items():
        all_scores = get_retrieval_scores_batched(model,
                                                  dataset,
                                                  tokenizer,
                                                  args.batch_size,
                                                  args.workers,
                                                  device)
        scores = dataset.evaluate_scores(all_scores)

        if dataset_name in ['coco_order', 'flickr30k_order']:
            metric_name = 'Precision@1'
            accuracy = scores[0][metric_name]
        else:
            symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing',
                         'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain',
                         'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down',
                         'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with',
                         'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over',
                         'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in',
                         'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off',
                         'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off',
                         'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving',
                         'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over',
                         'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on',
                         'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of',
                         'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in',
                         'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on',
                         'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading',
                         'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through',
                         'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near',
                         'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on',
                         'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against',
                         'standing around', 'standing behind', 'standing beside', 'standing in front of',
                         'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in',
                         'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying',
                         'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by',
                         'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on',
                         'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside',
                         'on the side of', 'around']
            df = pd.DataFrame(scores)
            if dataset_name == "vg_relation":
                df = df[~df.Relation.isin(symmetric)]
            elif dataset_name == "vg_attributes":
                df = df[~df.Attributes.isin(symmetric)]
            df = df[df["Count"] > 9]  # removing those with less than 9 counts
            accuracy = df.Accuracy.mean()
            metric_name = 'Macro Accuracy'

        logging.info(f"Eval Epoch {epoch - 1}: {dataset_name} accuracy: {accuracy:.4f}")
        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb.log({f"val/{dataset_name}-{metric_name}": accuracy, 'epoch': epoch})
