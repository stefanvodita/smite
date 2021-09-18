import cv2
import random
import numpy as np
import os


class Primus:
    def __init__(self, corpus_dirpath, trainset_filepath, vocabulary_filepath, is_semantic,
                 is_distorted = False, test_ratio = 0.0, separator = "\t", rnd_seed=None):
        self.separator = separator
        self.is_semantic = is_semantic
        self.is_distorted = is_distorted
        self.corpus_dirpath = corpus_dirpath

        # Vocabulary
        with open(vocabulary_filepath, "r") as vocabulary_file:
            self.vocabulary_list = vocabulary_file.read().splitlines()
        self.vocabulary_dict = {v: i for i, v in enumerate(self.vocabulary_list)}
        self.vocabulary_size = len(self.vocabulary_list)

        # Special symbols
        self.clef_idxs = [i for i, v in enumerate(self.vocabulary_list) if "clef" in v]
        self.key_sign_idxs = [i for i, v in enumerate(self.vocabulary_list) if "keySignature" in v]
        self.time_sign_idxs = [i for i, v in enumerate(self.vocabulary_list) if "timeSignature" in v]

        # Trainset entries
        with open(trainset_filepath, "r") as trainset_file:
            trainset_list = trainset_file.read().splitlines()

        # Train and validation split
        if rnd_seed is not None:
            random.seed(rnd_seed)
            random.shuffle(trainset_list)
        idx = int(len(trainset_list) * (1 - test_ratio))
        self.training_list = trainset_list[:idx]
        self.validation_list = trainset_list[idx:]
        
        print("Training with " + str(len(self.training_list)) + " and validating with " + str(len(self.validation_list)))


    def resize(self, image, height):
        width = int(float(height * image.shape[1]) / image.shape[0])
        return cv2.resize(image, (width, height))


    def invert(self, image):
        return 255 - image


    def normalize(self, image):
        return (255. - image) / 255.


    def load_external(self, path, nth=1):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self.nth_img(img, nth)


    def load_sample(self, sample_name, img_height):
        sample_dirpath = os.path.join(self.corpus_dirpath, sample_name)

        img_path = os.path.join(sample_dirpath, sample_name + ("_distorted.jpg" if self.is_distorted else ".png"))
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        except:
            return None, None
        # img = self.resize(img, img_height)
        # img = self.invert(img)  # img = self.normalize(img)

        gt_path = os.path.join(sample_dirpath, sample_name + (".semantic" if self.is_semantic else ".agnostic"))
        with open(gt_path, "r") as gt_file:
            gt = gt_file.readline().rstrip().split(self.separator)
        labels =  [self.vocabulary_dict[x] for x in gt]

        return img, labels


    def record_sizes(self):
        heights = []
        widths = []
        for sample_name in (self.training_list  + self.validation_list):
            sample_dirpath = os.path.join(self.corpus_dirpath, sample_name)
            img_path = os.path.join(sample_dirpath, sample_name + ("_distorted.jpg" if self.is_distorted else ".png"))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            heights.append(img.shape[0])
            widths.append(img.shape[1])

        with open("heights.txt", "w") as f:
            f.write(str(heights))
        with open("widths.txt", "w") as f:
            f.write(str(widths))


    def write_res(self, sample_name, prediction, acc):
        sample_dirpath = os.path.join(self.corpus_dirpath, sample_name)
        pred_path = os.path.join(sample_dirpath, sample_name + ".pred")

        # print(pred_path)

        with open(pred_path, "w") as f:
            f.write(str(acc) + "\n")
            f.write("\n".join([self.vocabulary_list[i] if i < len(self.vocabulary_list) else "stop" for i in prediction]))


    def transpose_img(self, image):
        return list(map(list, zip(*image)))


    def pad_zero_cols(self, col, width):
        return [[0 for _ in range(len(col))] for _ in range(width - 1)] + [col]


    def find_staff(self, image, threshold=5):
        for i, col in enumerate(image.T):
            for j in range(i, i + threshold):
                if not np.array_equal(col, image.T[j, :]):
                    break
            else:
                return col


    def find_clean_nth_idxs(self, image, n):
        nth = image.shape[1] // n
        staff = self.find_staff(image)
        idxs = [0]

        # print(staff)

        for i in range(1, n):
            idx = i * nth
            while not np.array_equal(np.where(image[:, idx] != 0), np.where(staff != 0)):
                # print(staff)
                # print(image[:, idx])
                idx -= 1
                if idx < 0:
                    # fall back to standard split
                    return [j * nth for j in range(n)] + [image.shape[1]]
            # print("staff match")
            idxs.append(idx)

        idxs.append(image.shape[1])
        return staff, idxs


    """
    Assumes inverted image
    """
    def clean_nth_img(self, image, n, pad="staff"):
        staff, idxs = self.find_clean_nth_idxs(image, n)
        # print(staff.shape)
        # print(idxs)
        nths = [image[:, idxs[i]:idxs[i+1]] for i in range(n)]
        maxlen = max([x.shape[1] for x in nths])
        if pad != "staff":
            nths = list(map(lambda x: np.hstack((x, [[0 for _ in range(maxlen - x.shape[1])] for _ in range(x.shape[0])])) \
                                if x.shape[1] != maxlen else x, nths))
        else:
            nths = list(map(lambda x: np.hstack((x, np.column_stack(list(staff for _ in range(maxlen - x.shape[1]))))) \
                                if x.shape[1] != maxlen else x, nths))
        return np.vstack(nths)


    def halve_img(self, image):
        half = image.shape[1] // 2
        return np.vstack((image[:, :half], image[:, half:2*half]))


    def third_img(self, image):
        third = image.shape[1] // 3
        return np.vstack((image[:, :third], image[:, third:2*third], image[:, 2*third:3*third]))


    def nth_img(self, image, n):
        nth = image.shape[1] // n
        return np.vstack(list([image[:, i * nth : (i + 1) * nth] for i in range(n)]))


    def load_batch(self, idx, count, img_height, is_training=True, nth=1):
        images = []
        labels = []
        for i in range(idx, idx + count):
            if is_training:
                if i >= len(self.training_list):
                    return images, labels, i - idx
                img, label = self.load_sample(self.training_list[i], img_height)
            else:
                if i >= len(self.validation_list):
                    return images, labels, i - idx
                img, label = self.load_sample(self.validation_list[i], img_height)

            # uncommented version is recommended, but the others work too
            # images.append(img)
            # images.append(self.halve_img(img))
            # images.append(self.third_img(img))
            images.append(self.nth_img(img, nth))
            # images.append(self.clean_nth_img(img, nth, pad="staff"))

            labels.append(label)
        return images, labels, count


    def write_results(self, predictions, accs, idx, is_training=True):
        for i in range(len(accs)):
            if is_training:
                self.write_res(self.training_list[idx + i], predictions[i], accs[i])
            else:
                self.write_res(self.validation_list[idx + i], predictions[i], accs[i])
