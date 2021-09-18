import argparse
import collections
import numpy as np
import sys
import timeit
import torch

from model import ModViT
from polyleven import levenshtein
from primus import Primus
from transformers import AdamW, PreTrainedModel, ViTConfig, ViTFeatureExtractor, ViTModel
from statistics import mean


# Parameterization
PATH = "./model"

EPOCHS = 5
BATCH_SIZE = 5000   # should go up to 23400
SET_SIZE = 70000

LEARNING_RATE = 5e-5

IMG_HEIGHT = 112 # 128 # 224

NUM_TOKENS = 60

VIT_LINES = 197
VIT_COLS = 768
"""
768, vit-base
1024, vit-large
"""

NTH = 5


parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
args = parser.parse_args()


def get_predicted_caption(prediction, special_bias=False):
    if not special_bias:
        # don't prefer certain symbols, like clefs or signatures
        return [torch.argmax(p).item() for p in prediction]

    # should be a clef
    first = primus.clef_idxs[torch.argmax(prediction[0][primus.clef_idxs])]

    # should be a key sign or a time sign
    second_idx = torch.argmax(prediction[1][primus.key_sign_idxs + primus.time_sign_idxs])
    if second_idx < len(primus.key_sign_idxs):
        second = primus.key_sign_idxs[second_idx]
    else:
        second = primus.time_sign_idxs[second_idx - len(primus.key_sign_idxs)]

    caption = [first, second]
    caption += [torch.argmax(p).item() for p in prediction[2:]]
    return caption


def accuracy(output, caption):
    return sum([1 if p == k else 0 for p, k in zip(get_predicted_caption(output), caption)]) / caption.shape[0]


def top_n_accuracy(output, caption, n=1):
    return sum([1 if k in torch.topk(p, n).indices else 0 for p, k in zip(output, caption)]) / caption.shape[0]


def edit_dist(output, caption):
    pred = get_predicted_caption(output)
    try:
        end_idx = pred.index(primus.vocabulary_size)
    except:
        end_idx = len(pred)
    a = "".join(map(chr, pred[:end_idx]))
    b = "".join(map(chr, caption[:-1]))
    return levenshtein(a, b) / len(b)


def simple_pass_through(img, gpu=True):
    img = torch.as_tensor([img, img, img])  # img.shape = torch.Size([3, IMG_HEIGHT, img_width])

    inputs = feature_extractor(images=img, return_tensors="pt")
    inputs = inputs.to("cuda" if gpu else "cpu")  # inputs["pixel_values"].shape = torch.Size([1, 3, IMG_HEIGHT, IMG_HEIGHT])

    out = model(inputs)

    return out


def pass_through(img, caption, losses, acc, leven):
    img = torch.as_tensor([img, img, img])  # img.shape = torch.Size([3, IMG_HEIGHT, img_width])

    inputs = feature_extractor(images=img, return_tensors="pt")
    inputs = inputs.to("cuda")  # inputs["pixel_values"].shape = torch.Size([1, 3, IMG_HEIGHT, IMG_HEIGHT])

    out = model(inputs)
    out = out[:len(caption) + 1]    # out.shape = torch.Size([caption_len + 1, primus.vocabulary_size + 1])

    caption.append(primus.vocabulary_size)
    caption = torch.as_tensor(caption)
    caption = caption.cuda()    # caption.shape = torch.Size([caption_len + 1])

    loss = loss_fn(out, caption)

    losses.append(loss.item())
    # acc.append(accuracy(out, caption))
    acc.append(top_n_accuracy(out, caption, n=5))
    leven.append(edit_dist(out, caption))

    return out, loss


def eval(idx, size, write_results=False):
    model.eval()

    predictions = []
    losses = []
    acc = []
    leven = []

    images, captions, count = primus.load_batch(idx, size, IMG_HEIGHT, is_training=False, nth=NTH)

    for img, caption in zip(images, captions):
        # skip exceptionaly long captions
        # if len(caption) > NUM_TOKENS - 1:
        #     continue

        out, _ = pass_through(img, caption, losses, acc, leven)
        predictions.append(out)

    if write_results:
        predictions = [get_predicted_caption(prediction) for prediction in predictions]
        primus.write_results(predictions, acc, idx, is_training=False)

    print("[EVAL] acc = {:.4f} leven = {:.4f}".format(mean(acc), mean(leven)), "loss =", mean(losses))

    model.train()
    return count


def compare_nths(idx, size, n):
    images, captions, _ = primus.load_batch(idx, size, IMG_HEIGHT, is_training=False, nth=1)

    for image, caption in zip(images, captions):
        print("Caption:\n", caption)

        for i in range(1, n + 1):
            nth = image.shape[1] // i
            imgs = list([image[:, j * nth : (j + 1) * nth] for j in range(i)])

            output = []
            for img in imgs:
                img = primus.nth_img(img, NTH)
                out = simple_pass_through(img)
                out = get_predicted_caption(out)
                output += out[:out.index(primus.vocabulary_size)]
                output.append(-1)

            print("Image spit into", i, "parts yields:\n", output)


def make_confusion_matrix(idx, size, training=False):
    confusion = np.zeros((primus.vocabulary_size + 1, primus.vocabulary_size + 1), dtype=int)
    freq = collections.Counter()

    images, captions, _ = primus.load_batch(idx, size, IMG_HEIGHT, is_training=training, nth=NTH)

    for img, caption in zip(images, captions):
        freq.update(caption)

        out = simple_pass_through(img)
        out = get_predicted_caption(out)

        for pred, tar in zip(out, caption):
            confusion[pred][tar] += 1
    return confusion, freq


def print_confusion_matrix(confusion, freq=None):
    confusion = [(tar, \
                    primus.vocabulary_list[tar] if tar < primus.vocabulary_size else "stop", \
                    pred, \
                    primus.vocabulary_list[pred] if pred < primus.vocabulary_size else "stop", \
                    confusion[pred][tar] / freq[tar] if freq else confusion[pred][tar], \
                    freq[tar] if freq else -1) \
                for pred in range(primus.vocabulary_size + 1) \
                for tar in range(primus.vocabulary_size + 1) \
                if pred != tar and confusion[pred][tar] != 0]
    confusion.sort(key=lambda x: x[4], reverse=True)
    confusion = list(map(str, confusion))
    print("\n".join(confusion))


def compute_inference_time_cpu():
    reps = 10
    warmup = 10
    timings=np.zeros((reps, 1))

    model.cpu()
    model.eval()
    with torch.no_grad():
        # warmup
        images, _, _ = primus.load_batch(0, warmup, IMG_HEIGHT, is_training=False, nth=NTH)
        for i in range(warmup):
             _ = simple_pass_through(images[i], gpu=False)

        # timed
        for i in range(reps):
            start = timeit.default_timer()

            images, _, _ = primus.load_batch(i, 1, IMG_HEIGHT, is_training=False, nth=NTH)
            out = simple_pass_through(images[0], gpu=False)
            get_predicted_caption(out)

            timings[i] = timeit.default_timer() - start
    model.train()

    mean_t, std_t = np.sum(timings) / reps, np.std(timings)
    print(mean_t, std_t)
    return mean_t, std_t


def compute_inference_time_gpu():
    """
    https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    """
    reps = 1000
    warmup = 10
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings=np.zeros((reps, 1))

    if not gpu:
        model.cpu()

    model.eval()
    with torch.no_grad():
        # warmup
        images, _, _ = primus.load_batch(0, warmup, IMG_HEIGHT, is_training=False, nth=NTH)
        for i in range(warmup):
             _ = simple_pass_through(images[i], gpu=True)

        # timed
        for i in range(reps):
            starter.record()

            images, _, _ = primus.load_batch(i, 1, IMG_HEIGHT, is_training=False, nth=NTH)
            out = simple_pass_through(images[0], gpu=True)
            get_predicted_caption(out)

            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)
    model.train()

    mean_t, std_t = np.sum(timings) / reps, np.std(timings)
    print(mean_t, std_t)
    return mean_t, std_t


def count_perfect_predictions(idx, size):
    model.eval()

    acc = []

    images, captions, _ = primus.load_batch(idx, size, IMG_HEIGHT, is_training=False, nth=NTH)

    for img, caption in zip(images, captions):
        # skip exceptionaly long captions
        if len(caption) > NUM_TOKENS - 1:
            continue

        out, _ = pass_through(img, caption, [], acc, [])

    # print(acc.index(next(x for x in acc if x < 0.5)))

    ct = acc.count(1)
    print("Perfect predictions: {}; out of: {}; proportion: {}".format(ct, len(acc), ct / len(acc)))

    model.train()


def infere_one(idx, is_training=True, write_to="prediction.txt"):
    print(primus.training_list[idx] if is_training else primus.validation_list[idx])
    print()

    images, captions, _ = primus.load_batch(idx, 1, IMG_HEIGHT, is_training=is_training, nth=NTH)
    out = simple_pass_through(images[0], gpu=False)

    pred = get_predicted_caption(out)
    print("Encoded prediction:\n{}\n".format(pred))

    pred = pred[:pred.index(primus.vocabulary_size)]
    print("Encoded cleaned prediction:\n{}\n".format(pred))

    pred = list(map(lambda x: primus.vocabulary_list[x], pred))
    print("Decoded prediction:\n{}\n".format(pred))

    if write_to:
        with open(write_to, "w") as f:
            f.write("\t".join(pred))

    tar = list(map(lambda x: primus.vocabulary_list[x], captions[0]))
    print("Target:\n{}\n".format(tar))

    acc = top_n_accuracy(out, torch.as_tensor(captions[0]), n=1)
    lev = edit_dist(out, torch.as_tensor(captions[0] + [0]))
    print("Accuracy:\t{}\nLevenshtein:\t{}\n".format(acc, lev))


def infere_external(path, write_to="prediction.txt"):
    image = primus.load_external(path, nth=5)
    out = simple_pass_through(image, gpu=False)

    pred = get_predicted_caption(out)
    print("Encoded prediction:\n{}\n".format(pred))

    pred = pred[:pred.index(primus.vocabulary_size)]
    print("Encoded cleaned prediction:\n{}\n".format(pred))

    pred = list(map(lambda x: primus.vocabulary_list[x], pred))
    print("Decoded prediction:\n{}\n".format(pred))

    if write_to:
        with open(write_to, "w") as f:
            f.write("\t".join(pred))


def get_target_sizes(write_to):
    sizes = collections.Counter()
    idx = 0
    while idx < 70000:
        _, captions, count = primus.load_batch(idx, 10000, IMG_HEIGHT, is_training=True, nth=1)
        idx += count
        sizes.update([len(x) for x in captions])
    
    with open(write_to, "w") as f:
        f.write(str(dict(sizes)))


def querry_target_sizes(read_from):
    from ast import literal_eval
    with open(read_from, "r") as f:
        sizes = literal_eval(f.read())

    maxlen = max(sizes.keys())
    n_big_lens = sum([x[1] for x in sizes.items() if x[0] >= NUM_TOKENS])

    print("Maximum sequence length:", maxlen)
    print("Number of lengths larger than output size:", n_big_lens)

    distrib = []
    for size, count in sizes.items():
        distrib += [size] * count

    print("Mean length:", mean(distrib))
    print("Standard deviation of lengths:", np.std(distrib))

    print(distrib)


def count_parameters(model):
    """
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


primus = Primus(args.corpus, args.set, args.voc, args.semantic, test_ratio = 0.1, rnd_seed=None)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

model = ModViT(VIT_COLS, primus.vocabulary_size + 1,
               VIT_LINES, NUM_TOKENS,
               pretrained_vit=True)

optim = AdamW(model.parameters(), lr=LEARNING_RATE)

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.cuda()

# model.load(PATH)  # implicit gpu=True
model.load(PATH, gpu=False)

# Bellow are various things you can do with this code:

# with torch.no_grad():
#     eval(0, 7000)

# compare_nths(0, 1, 2)

# confusion, freq = make_confusion_matrix(0, 50, training=False)
# print_confusion_matrix(confusion, freq)

# compute_inference_time_cpu()

# compute_inference_time_gpu()

# count_perfect_predictions(0, 40)

# eval(0, 50, write_results=True)

infere_one(26169)  # demo
# infere_one(31, is_training=False)  # good
# infere_one(27, is_training=False)  # inexplicable
# infere_one(64, is_training=False)  # notes moved 2 steps higher on the staff
# infere_one(76, is_training=False)  # repeated notes skipped

# get_target_sizes("sizes.txt")

# querry_target_sizes("records/dataset/sizes.txt")

# print(count_parameters(model))

# print(model)

# primus.record_sizes()

# infere_external("../samples/empire3.PNG")  # empire3, ussr2, dontstop
sys.exit(0)

# Or comment all above, and let flow get to here, where training will be done.
# Have a look at some of the training configurations, eg: stop conditions,
# periodic evaluation steps, and status report prints.

model.train()
for epoch in range(EPOCHS):
    sample_idx = 0  # index in the dataset of the sample to start train on
    losses = []
    acc = []
    leven = []
    stop = False

    while True:
        if sample_idx >= SET_SIZE:
            print("Reached set end")
            break

        # images = (count, IMG_HEIGHT, img_width)
        # captions = (count, caption_len)
        images, captions, count = primus.load_batch(sample_idx, BATCH_SIZE, IMG_HEIGHT, nth=NTH)

        if not images:
            print("Out of samples")
            break

        for img, caption in zip(images, captions):
            sample_idx += 1

            # skip exceptionaly long captions
            # if len(caption) > NUM_TOKENS - 1:
            #     continue

            optim.zero_grad()

            _, loss = pass_through(img, caption, losses, acc, leven)

            loss.backward()
            optim.step()

            if sample_idx % 5000 == 0:
                print("epoch =", epoch, "sample_idx =", sample_idx, \
                      "acc = {:.4f} leven = {:.4f}".format(mean(acc), mean(leven)), \
                      "loss =", mean(losses), flush=True)

                stop = True if stop or mean(losses) < 0.02 else False

                losses = []
                acc = []
                leven = []
            if sample_idx % 5000 == 0:
                with torch.no_grad():
                    eval(5000, 1000)
                    print(flush=True)
            if sample_idx % 5000 == 0:
                model.save(PATH)

            # loss.backward()
            # optim.step()
        # break   # only take 1 batch
    # break # only go for 1 epoch

    # with torch.no_grad():
    #     eval(0, 100)
    # model.save(PATH)
    # print(flush=True)

    if stop:
        print("Reached stop condition")
        break
