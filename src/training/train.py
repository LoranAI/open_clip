import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            # break
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                                         F.cross_entropy(logits_per_image, labels) +
                                         F.cross_entropy(logits_per_text, labels)
                                 ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


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


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


@torch.no_grad()
def get_retrieval_scores_batched(model,  # assuming to have a CLIPModel
                                 dataset,
                                 tokenizer,
                                 batch_size=128,
                                 num_workers=4,
                                 device="cuda"
                                 ):
    """
    from https://github.com/mertyg/vision-language-models-are-bows/blob/main/model_zoo/clip_models.py#L55
    Computes the scores for each image_option / caption_option pair in the joint loader.

    Args:
        joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
        "image_options" is a list of images, and "caption_options" is a list of captions.

    Returns:
        all_scores: A numpy array containing the scores of the shape NxKxL,
        where N is the number of test cases, K is the number of image options per the test case,
        and L is the number of caption options per the test case.
    """
    scores = []
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    tqdm_loader = tqdm(data_loader)
    tqdm_loader.set_description("Computing retrieval scores")
    for batch in tqdm_loader:
        image_options = []
        for i_option in batch["image_options"]:
            # i_option = torch.cat(i_option.pixel_values)
            # i_option = torch.cat(i_option)
            image_embeddings = model.encode_image(i_option.to(device)).cpu().numpy()  # B x D
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # B x D
            image_options.append(np.expand_dims(image_embeddings, axis=1))

        caption_options = []
        for c_option in batch["caption_options"]:
            # caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
            caption_tokenized = tokenizer(c_option)
            caption_embeddings = model.encode_text(caption_tokenized.to(device)).cpu().numpy()  # B x D
            caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)  # B x D
            caption_options.append(np.expand_dims(caption_embeddings, axis=1))

        image_options = np.concatenate(image_options, axis=1)  # B x K x D
        caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
        batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L
        scores.append(batch_scores)

    all_scores = np.concatenate(scores, axis=0)  # N x K x L
    return all_scores


# FIXME Check if working
def evaluate_VAL(model, data, tokenizer, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    dataset = data["val"]
    all_scores = get_retrieval_scores_batched_VAL(model,
                                                  dataset,
                                                  device)
    scores = evaluate_scores_VAL(all_scores)

    metric_name = 'Precision@1'
    accuracy = scores[0][metric_name]

    logging.info(f"Eval Epoch {epoch - 1}: accuracy: {accuracy:.4f}")
    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        wandb.log({f"val/{metric_name}": accuracy, 'epoch': epoch})


@torch.no_grad()
def get_retrieval_scores_batched_VAL(model,  # assuming to have a CLIPModel
                                     dataset,
                                     device="cuda"
                                     ):
    """
    from https://github.com/mertyg/vision-language-models-are-bows/blob/main/model_zoo/clip_models.py#L55
    Computes the scores for each image_option / caption_option pair in the joint loader.

    Args:
        joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
        "image_options" is a list of images, and "caption_options" is a list of captions.

    Returns:
        all_scores: A numpy array containing the scores of the shape NxKxL,
        where N is the number of test cases, K is the number of image options per the test case,
        and L is the number of caption options per the test case.
    """
    scores = []
    data_loader = dataset.dataloader
    tqdm_loader = tqdm(data_loader)
    tqdm_loader.set_description("Computing retrieval scores")
    for batch in tqdm_loader:
        image_options = []
        i_option = batch[0]
        # i_option = torch.cat(i_option.pixel_values)
        # i_option = torch.cat(i_option)
        image_embeddings = model.encode_image(i_option.to(device)).cpu().numpy()  # B x D
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # B x D
        image_options.append(np.expand_dims(image_embeddings, axis=1))

        caption_options = []
        c_option = batch[1]
        # caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
        caption_tokenized = c_option
        caption_embeddings = model.encode_text(caption_tokenized.to(device)).cpu().numpy()  # B x D
        caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)  # B x D
        caption_options.append(np.expand_dims(caption_embeddings, axis=1))

        image_options = np.concatenate(image_options, axis=1)  # B x K x D
        caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
        batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L
        scores.append(batch_scores)

    all_scores = np.concatenate(scores, axis=0)  # N x K x L
    return all_scores


def evaluate_scores_VAL(scores):

    if isinstance(scores, tuple):
        scores_i2t = scores[0]
        scores_t2i = scores[1].T  # Make it N_ims x N_text

    else:
        scores_t2i = scores
        scores_i2t = scores

    preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
    correct_mask = (preds == 0)
    records = [{"Precision@1": np.mean(correct_mask)}]
    for k in [1, 5, 10]:
        stringss = f"RECALL@{k}: {np.mean(preds < k)}"
        print(stringss)
    return records
