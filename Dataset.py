import json

import cv2

from config import cfg
import numpy as np
import utils as utils


class Dataset(object):
    def __init__(self, dataset_type):
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TRAIN.ANNOT_PATH
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        # self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_size = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = np.array(utils.get_anchors())
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.img_id_list = self.load_car_person_list()[
                           :1000] if dataset_type == 'train' else self.load_car_person_list()[1000:1200]
        self.num_samples = len(self.img_id_list)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # self.train_input_size = random.choice(self.train_input_size)
        self.train_output_sizes = self.train_input_size // self.strides
        batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

        # label_bbox
        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes))

        # bbox
        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

        num = 0
        if self.batch_count < self.num_batches:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: index -= self.num_samples
                id = self.img_id_list[index]
                image, bboxes = self.parse_annotations(self.annotations, id)
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                batch_image[num, :, :, :] = image
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 1
            self.batch_count += 1
            return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes

        else:
            self.batch_count = 0
            # np.random.shuffle(self.annotations)
            raise StopIteration

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            annotations = json.load(f)
        # np.random.shuffle(annotations)
        return annotations

    def load_car_person_list(self):
        open_file = open('car_and_person.txt').read()
        id_str = ""
        for line in open_file:
            id_str += line

        id_str = id_str.strip("[]").split(", ")

        for i in range(0, len(id_str)):
            id_str[i] = int(id_str[i])
        print(type(id_str))
        print(id_str)
        return id_str

    def parse_annotations(self, annotation, id):
        image_path = '/Users/JTSAI1/Documents/ADAI/train_car_person/' + str(id) + '.jpg'
        image = np.array(cv2.imread(image_path))
        bboxes = []

        for ann in annotation['annotations']:
            if ann['image_id'] == id:
                x = ann['bbox'][0]
                y = ann['bbox'][1]
                w = ann['bbox'][2]
                h = ann['bbox'][3]
                c = ann['category_id']
                if c != 1 and c != 3:
                    continue
                elif c == 1:
                    c = 0
                else:
                    c = 1
                x, y, w, h = int(x), int(y), int(w), int(h)
                bboxes.append([x, y, w, h, c])
                # print([x, y, w, h, c])

        bboxes = np.array(bboxes)
        # print("bboxes.shape Dataset: ", bboxes.shape)
        # print("bboxes:\n", bboxes)
        image, bboxes = utils.image_preporcess(image, [self.train_input_size, self.train_input_size],
                                               np.copy(bboxes))
        # print("bboxes after processed:\n", bboxes)
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            # print("bbox_coor: ", bbox_coor)
            # print("bbox_class_ind: ", bbox_class_ind)

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # print("(bbox_coor[2:] + bbox_coor[:2]): ", (bbox_coor[2:] + bbox_coor[:2]))

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)
            # print("self.strides: ", self.strides)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # print("best_detect: ", best_detect)
                # print("xind, yind: ", xind, yind)
                # print("best_anchor: ", best_anchor)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batches

# if __name__ == "__main__":
#     dataset = Dataset('train')
#     dataset.__next__()
# dataset.parse_annotations(dataset.annotations)
# print(dataset.img_id)
