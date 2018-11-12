# coding:utf-8

import copy
import os
import pickle
import shutil
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import layers as tl
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from seaborn import kdeplot


class Config:
    """
    x axis : ↓
    y axis : →
    proposal : [center_x, center_y, height, width]
    anchors : [height, width]
    shape : [height, width]
    """

    def __init__(self, image_shape, channel, sample_shape, anchors, fg_thresh=0.65, bg_thresh=0.3, floating=5):
        self.image_shape = image_shape
        self.channel = channel
        self.sample_shape = sample_shape
        self.anchors = np.array(anchors)
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.scale = (self.image_shape[0] / sample_shape[0], self.image_shape[1] / sample_shape[1])
        self.sample_size = len(self.anchors) * sample_shape[0] * sample_shape[1]
        self.proposal = np.zeros((self.sample_size, 4))
        self.proposal[:, 0] = np.repeat(np.arange(sample_shape[0]) * self.scale[0] + self.scale[0] // 2, sample_shape[1] * len(self.anchors))
        self.proposal[:, 1] = np.tile(np.repeat(np.arange(sample_shape[1]) * self.scale[1] + self.scale[0] // 2, len(self.anchors)), sample_shape[0])
        self.proposal[:, 2:] = np.tile(self.anchors.transpose(), sample_shape[0] * sample_shape[1]).transpose()
        self.bbox = np.zeros_like(self.proposal)
        self.bbox[:, 0] = self.proposal[:, 0] - self.proposal[:, 2] / 2
        self.bbox[:, 1] = self.proposal[:, 1] - self.proposal[:, 3] / 2
        self.bbox[:, 2] = self.proposal[:, 0] + self.proposal[:, 2] / 2
        self.bbox[:, 3] = self.proposal[:, 1] + self.proposal[:, 3] / 2
        self.proposal_mask = np.logical_and(
            np.logical_and(self.bbox[:, 0] >= -floating, self.bbox[:, 2] <= self.image_shape[0] + floating),
            np.logical_and(self.bbox[:, 1] >= -floating, self.bbox[:, 3] <= self.image_shape[1] + floating)
        ).astype(np.float32)


class Imdb:
    def __init__(self, path):
        self.path = path
        self.image = None
        self.bbox = None
        self.ground_true_box = None
        self.ignore = None
        self.kind = None
        self.proposal_regress = None
        self.proposal_regress_mask = None


class DataEngineBase(metaclass=ABCMeta):
    def __init__(self, config: Config, keep_size=256):
        self.imdbs = []
        self.config = config
        self.full_kind_encoder = {}
        self.keep_kind_encoder = {}
        self.keep_size = keep_size

    @abstractmethod
    def load_origin_data(self, **kwargs):
        pass

    def load_info(self, info_path):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
            self.full_kind_encoder = info["full_kind_encoder"]
            self.keep_kind_encoder = info["keep_kind_encoder"]

    def resize_crop_image(self, image: Image, gtb: np.ndarray):
        output_shape = self.config.sample_shape
        input_shape = image.size
        if input_shape[0] / output_shape[0] < input_shape[1] / output_shape[1]:
            scale = input_shape[0] / output_shape[0]
            image = image.resize((output_shape[0], np.round(input_shape[1] / scale).astype(np.int)))
        else:
            scale = input_shape[1] / output_shape[1]
            image = image.resize((np.round(input_shape[0] / scale).astype(np.int), output_shape[1]))
        gtb = np.divide(gtb, scale)
        gtb_border = (
            np.min(gtb[:, 0] - gtb[:, 2] / 2),
            np.min(gtb[:, 1] - gtb[:, 3] / 2),
            np.max(gtb[:, 0] + gtb[:, 2] / 2),
            np.max(gtb[:, 1] + gtb[:, 3] / 2)
        )
        gtb_center = (
            (gtb_border[0] + gtb_border[2]) / 2,
            (gtb_border[1] + gtb_border[3]) / 2
        )
        border = [
            max(0, np.round(gtb_center[0] - gtb_center[0] / 2).astype(np.int)),
            max(0, np.round(gtb_center[1] - gtb_center[1] / 2).astype(np.int)),
            min(image.size[0], np.round(gtb_center[0] + gtb_center[0] / 2).astype(np.int)),
            min(image.size[1], np.round(gtb_center[1] + gtb_center[1] / 2).astype(np.int)),
        ]
        if border[2] - border[0] != output_shape[0]:
            if border[0] + output_shape[0] <= image.size[0]:
                border[2] = border[0] + output_shape[0]
            elif border[2] - output_shape[0] >= 0:
                border[0] = border[2] - output_shape[0]
            else:
                raise ValueError
        if border[3] - border[1] != output_shape[1]:
            if border[1] + output_shape[1] <= image.size[1]:
                border[3] = border[1] + output_shape[1]
            elif border[3] - output_shape[1] >= 0:
                border[1] = border[3] - output_shape[1]
            else:
                raise ValueError
        image = image.crop(border)
        gtb[:, 0:2] -= border[0:2]
        return image, gtb

    @abstractmethod
    def filter_func(self, imdb: Imdb):
        return imdb.ignore

    def filter_gtb(self, filter_func):
        if not callable(filter_func):
            raise TypeError("filter_func must be callable, filter_func(imdb: Imdb)")
        for imdb in self.imdbs:
            imdb.ignore = filter_func(imdb)
            for kind, ignore in zip(imdb.ignore, imdb.kind):
                if ignore == 0 and self.keep_kind_encoder.get(kind, None) is None:
                    self.keep_kind_encoder[kind] = len(self.keep_kind_encoder)

    def convert_code(self, code):
        return self.keep_kind_encoder[code]

    @staticmethod
    def write_imdb_batch(imdbs, directory, index):
        filename = "%06d.tfrecords" % index
        writer = tf.python_io.TFRecordWriter(os.path.join(directory, filename))
        for imdb in imdbs:
            item = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imdb.image.tobytes()])),
                        "kind": tf.train.Feature(int64_list=tf.train.Int64List(value=imdb.kind)),
                        "bbox": tf.train.Feature(float_list=tf.train.FloatList(value=imdb.bbox.flatten())),
                        "gtb": tf.train.Feature(float_list=tf.train.FloatList(value=imdb.ground_true_box.flatten())),
                        "ignore": tf.train.FloatList(int64_list=tf.train.Int64List(value=imdb.ignore)),
                        "count": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(imdb.ground_true_box)]))
                    }
                )
            )
            writer.write(item.SerializeToString())
        writer.close()
        print("write %s finished" % filename)

    def program(self, imdbs):
        for imdb in imdbs:
            imdb.image = Image.open(imdb.path)
            imdb.image, imdb.ground_true_box = self.resize_crop_image(imdb.image, imdb.ground_true_box)
        return imdbs

    def write_tfrecords(self, batch_size, directory):
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        os.mkdir(directory)
        for i in range(0, len(self.imdbs), batch_size):
            batch_imdbs = copy.deepcopy(self.imdbs[i:i + batch_size])
            batch_imdbs = self.program(batch_imdbs)
            self.write_imdb_batch(batch_imdbs, "data/train", i // batch_size)

    def parse_data(self, proto):
        features = tf.parse_single_example(
            proto, features={
                "image": tf.FixedLenFeature([], tf.string),
                "kind": tf.VarLenFeature(tf.int64),
                "bbox": tf.VarLenFeature(tf.float32),
                "gtb": tf.VarLenFeature(tf.float32),
                "ignore": tf.VarLenFeature(tf.int64),
                "count": tf.FixedLenFeature([1], tf.int64)
            }
        )
        image = tf.reshape(tf.decode_raw(features["image"], tf.uint8), (self.config.image_shape[0], self.config.image_shape[1], 3))
        count = features["count"]
        kind = tf.reshape(tf.sparse_tensor_to_dense(features["kind"]), count)
        bbox = tf.reshape(tf.sparse_tensor_to_dense(features["bbox"]), tf.concat([count, tf.constant([4], tf.int64)], 0))
        gtb = tf.reshape(tf.sparse_tensor_to_dense(features["gtb"]), tf.concat([count, tf.constant([4], tf.int64)], 0))
        ignore = tf.reshape(tf.sparse_tensor_to_dense(features["ignore"]), count)
        return {"image": image, "count": count, "kind": kind, "bbox": bbox, "gtb": gtb, "ignore": ignore}

    def read_tfrecords(self, directory):
        filename_list = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".tfrecords")]
        dataset = tf.data.TFRecordDataset(filename_list).map(self.parse_data).shuffle(2048).repeat()
        return dataset

    def calc_target(self, kind_code, gtb, ignore, regress_scale):
        kind_code = kind_code[ignore == 0]
        gtb = gtb[ignore == 0]
        overlap = self.calc_overlap(self.config.proposal, gtb)
        _, _, _, label, label_weight, proposal_regress, proposal_regress_weight = self.calc_label_regression(overlap, gtb, kind_code)
        return kind_code, label, label_weight, proposal_regress * regress_scale, proposal_regress_weight

    @staticmethod
    def calc_overlap(mat_1, mat_2):
        m_l = np.max(np.meshgrid(mat_1[:, 0] - mat_1[:, 2] / 2, mat_2[:, 0] - mat_2[:, 2] / 2, sparse=False, indexing='ij'), axis=0)
        m_u = np.max(np.meshgrid(mat_1[:, 1] - mat_1[:, 3] / 2, mat_2[:, 1] - mat_2[:, 3] / 2, sparse=False, indexing='ij'), axis=0)
        m_r = np.min(np.meshgrid(mat_1[:, 0] + mat_1[:, 2] / 2, mat_2[:, 0] + mat_2[:, 2] / 2, sparse=False, indexing='ij'), axis=0)
        m_d = np.min(np.meshgrid(mat_1[:, 1] + mat_1[:, 3] / 2, mat_2[:, 1] + mat_2[:, 3] / 2, sparse=False, indexing='ij'), axis=0)
        area_1 = mat_1[:, 2] * mat_1[:, 3]
        area_2 = mat_2[:, 2] * mat_2[:, 3]
        area_inner = np.maximum(m_d - m_u, 0) * np.maximum(m_r - m_l, 0)
        overlap = area_inner / (np.sum(np.meshgrid(area_1, area_2, sparse=False, indexing="ij"), axis=0) - area_inner)
        return overlap

    def calc_label_regression(self, overlap, roi, type_code):
        overlap_max = np.max(overlap, axis=1)
        choice_index = np.argmax(overlap, axis=1)
        fg_mask = np.where(overlap_max >= self.config.fg_thresh, 1, 0)
        bg_mask = np.where(overlap_max <= self.config.bg_thresh, 1, 0)
        binary_label = np.vstack([fg_mask, bg_mask]).T
        gtb_argmax = np.argmax(overlap, axis=0)
        binary_label[gtb_argmax] = [1, 0]
        choice_index[gtb_argmax] = np.arange(len(gtb_argmax))
        label = np.zeros((self.config.sample_size, len(self.keep_kind_encoder) + 1))
        label[np.arange(self.config.sample_size), np.array(list(map(lambda x: self.keep_kind_encoder[x], type_code[choice_index])))] = 1.0
        label[np.where(binary_label[:, 0]) == 0, :] = 0.
        label[np.where(binary_label[:, 1]) == 1, -1] = 1.
        choice_roi = roi[choice_index]
        proposal_regress = np.zeros((self.config.sample_size, 4))
        proposal_regress[:, 0] = (choice_roi[:, 0] - self.config.proposal[:, 0]) / self.config.proposal[:, 2]
        proposal_regress[:, 1] = (choice_roi[:, 1] - self.config.proposal[:, 1]) / self.config.proposal[:, 3]
        proposal_regress[:, 2] = np.log(choice_roi[:, 2] / self.config.proposal[:, 2])
        proposal_regress[:, 3] = np.log(choice_roi[:, 3] / self.config.proposal[:, 3])
        fg_index = np.where(binary_label[:, 0] * self.config.proposal_mask == 1)[0]
        bg_index = np.where(binary_label[:, 1] * self.config.proposal_mask == 1)[0]
        fg_len = len(fg_index)
        bg_len = len(bg_index)
        np.random.shuffle(fg_index)
        np.random.shuffle(bg_index)
        fg_count = min(fg_index.shape[0], self.keep_size // 2)
        bg_count = self.keep_size - fg_count
        label_weight = np.zeros(self.config.sample_size)
        proposal_regress_weight = np.zeros(self.config.sample_size)
        label_weight[fg_index[:fg_count]] = 1.
        label_weight[bg_index[:bg_count]] = 1.
        proposal_regress_weight[fg_index[:fg_count]] = 1.
        return fg_len, bg_len, binary_label, label, label_weight, proposal_regress, proposal_regress_weight


class Evaluation:
    def __init__(self, gtb_filename="gtb_feature.png", anchors_filename="fg&bg.png"):
        self.gtb_filename = gtb_filename
        self.anchors_filename = anchors_filename

    def count_ground_true_box(self, gtb: np.ndarray):
        plt.figure(figsize=(18, 6), dpi=200)
        plt.subplot(1, 2, 1)
        kdeplot(gtb[:, 0], gtb[:, 1], shade=True, cbar=True).set_title('xy')
        plt.subplot(1, 2, 2)
        kdeplot(gtb[:, 2], gtb[:, 3], shade=True, cbar=True).set_title('wh')
        plt.savefig(self.gtb_filename)
        plt.close()

    def evaluate_anchors(self, data_engine, imdbs: list, batch_size=256):
        fg = []
        bg = []
        for i in range(0, len(imdbs), batch_size):
            print("%d/%d" % (i, len(imdbs)))
            batch_imdbs = copy.deepcopy(data_engine.imdbs[i:i + batch_size])
            batch_imdbs = data_engine.program(batch_imdbs)
            for imdb in batch_imdbs:
                batch_imdbs = data_engine.program(batch_imdbs)
                if len(imdb.label) == 0:
                    continue
                kind = imdb.kind[imdb.ignore == 0]
                ground_true_box = imdb.ground_true_box[imdb.ignore == 0]
                overlap = data_engine.calc_overlap(data_engine.config.proposal, ground_true_box)
                fg_len, bg_len, _, _, _, _, _ = data_engine.calc_label_regression(overlap, ground_true_box, kind)
                fg.append(fg_len)
                bg.append(bg_len)
        plt.figure(figsize=(8, 6), dpi=200)
        kdeplot(fg, color="r")
        kdeplot(bg, color="b")
        plt.savefig(self.anchors_filename)
        plt.close()


class Visualization:
    def __init__(self, config, regress_scale, directory):
        self.config = config
        self.regress_scale = regress_scale
        self.directory = directory

    def function_nms(self, label, proposal, threshold=0.001, nt=0.3, method=None, sigma=0.5):
        gtb_regress = np.zeros((self.config.sample_size, 4))
        proposal /= self.regress_scale
        gtb_regress[:, 0] = proposal[:, 0] * self.config.proposal[:, 2] + self.config.proposal[:, 0]
        gtb_regress[:, 1] = proposal[:, 1] * self.config.proposal[:, 3] + self.config.proposal[:, 1]
        gtb_regress[:, 2] = np.exp(proposal[:, 2]) * self.config.proposal[:, 2]
        gtb_regress[:, 3] = np.exp(proposal[:, 3]) * self.config.proposal[:, 3]
        bbox = np.zeros((self.config.sample_size, 6))
        bbox[:, 0] = gtb_regress[:, 0] - gtb_regress[:, 2] / 2.
        bbox[:, 1] = gtb_regress[:, 1] - gtb_regress[:, 3] / 2.
        bbox[:, 2] = gtb_regress[:, 0] + gtb_regress[:, 2] / 2.
        bbox[:, 3] = gtb_regress[:, 1] + gtb_regress[:, 3] / 2.
        bbox[:, 4] = label[:, 0]
        bbox[:, 5] = label[:, 0]
        bbox = bbox[np.argsort(bbox[:, 4])[::-1], :]
        rest = self.config.sample_size
        keep_size = 0
        while keep_size < rest:
            areas = (bbox[keep_size:rest, 2] - bbox[keep_size:rest, 0]) * (bbox[keep_size:rest, 3] - bbox[keep_size:rest, 1])
            _x1 = np.maximum(bbox[keep_size + 1:rest, 0], bbox[keep_size, 0])
            _y1 = np.maximum(bbox[keep_size + 1:rest, 1], bbox[keep_size, 1])
            _x2 = np.minimum(bbox[keep_size + 1:rest, 2], bbox[keep_size, 2])
            _y2 = np.minimum(bbox[keep_size + 1:rest, 3], bbox[keep_size, 3])
            _w = np.maximum(_x2 - _x1, 0)
            _h = np.maximum(_y2 - _y1, 0)
            inter = _w * _h
            overlap = inter / (areas[0] + areas[1:] - inter)
            if method is "linear":
                weight = np.where(overlap > nt, 1.0 - overlap, 1.0)
            elif method is "gaussian":
                weight = np.exp(-np.power(overlap, 2) / sigma)
            else:
                weight = np.where(overlap > nt, 0.0, 1.0)
            keep_size += 1
            bbox[keep_size:rest, 4] = bbox[keep_size:rest, 4] * weight
            index = np.argsort(bbox[keep_size:rest, 4])[::-1] + keep_size
            bbox[keep_size:rest, :] = bbox[index, :]
            rest -= np.count_nonzero(bbox[keep_size:rest, 4] < threshold)
        return np.array(bbox[:keep_size, :4]), np.array(bbox[:keep_size, 5])

    @staticmethod
    def generate_image(image, bbox=None, pre_bbox=None, pre_score=None, threshold=0.99):
        if bbox is None:
            bbox = []
        if pre_bbox is None:
            pre_bbox = []
        if pre_score is None:
            pre_score = []
        draw = ImageDraw.Draw(image)
        for b in bbox:
            draw.rectangle(b.tolist(), outline="#00FF00")
        for b, s in zip(pre_bbox, pre_score):
            if s >= threshold:
                draw.rectangle(b.tolist(), outline="#FF0000")
        return image

    def save_image(self, image_data, filename, marked_bbox=None, marked_label=None, marked_proposal_regress=None, predict_label=None, predict_proposal_regress=None):
        image = Image.fromarray(image_data)
        draw = ImageDraw.Draw(image)
        if marked_bbox is not None:
            for bbox in marked_bbox:
                draw.rectangle(bbox.tolist(), outline="#00FF00")
        elif marked_label is not None and marked_proposal_regress is not None:
            marked_bbox, marked_score = self.function_nms(marked_label, marked_proposal_regress)
            for bbox in marked_bbox:
                draw.rectangle(bbox.tolist(), outline="#00FF00")
        if predict_label is not None and predict_proposal_regress is not None:
            predict_bbox, predict_score = self.function_nms(predict_label, predict_proposal_regress)
            for bbox in predict_bbox:
                draw.rectangle(bbox.tolist(), outline="#FF0000")
        image.save(os.path.join(self.directory, filename))

    def visualization(self, step, batch_image_data, batch_marked_bbox=None, batch_marked_label=None, batch_marked_proposal_regress=None, batch_predict_label=None, batch_predict_proposal_regress=None):
        for i, image_data in enumerate(batch_image_data):
            filename = "%06d_%03d.jpg" % (step, i)
            marked_bbox = batch_marked_bbox[i] if batch_marked_bbox is not None else None
            marked_label = batch_marked_label[i] if batch_marked_label is not None else None
            marked_proposal_regress = batch_marked_proposal_regress[i] if batch_marked_proposal_regress is not None else None
            predict_label = batch_predict_label[i] if batch_predict_label is not None else None
            predict_proposal_regress = batch_predict_proposal_regress[i] if batch_predict_proposal_regress is not None else None
            self.save_image(filename, marked_bbox, marked_label, marked_proposal_regress, predict_label, predict_proposal_regress)


class SpecialLayer:
    @staticmethod
    def test():
        x = tf.placeholder(tf.float32, [2, 8, 10, 3])
        batch_rois = np.array([
            [
                [-1, 1, 5, 4],
                [1, -1, 7, 5],
                [5, 7, 7, 10]
            ],
            [
                [-1, 1, 5, 4],
                [1, -1, 7, 5],
                [5, 7, 7, 10]
            ]
        ])
        shape = (2, 2)
        result = SpecialLayer.max_roi_pooling(x, batch_rois, shape)
        print(result.shape)

    @staticmethod
    def max_roi_pooling(x, rois, shape):
        x_shape = x.get_shape().as_list()
        rois[:, :, 0] = np.maximum(rois[:, :, 0], 0)
        rois[:, :, 1] = np.maximum(rois[:, :, 1], 0)
        rois[:, :, 2] = np.minimum(rois[:, :, 2], x_shape[1])
        rois[:, :, 3] = np.minimum(rois[:, :, 3], x_shape[2])
        rois = np.round(rois).astype(np.int)
        batch_size = len(rois)
        batch_output = []
        for value in tf.split(x, batch_size, 0):
            output = []
            for roi in rois:
                pos0 = np.round(np.linspace(roi[0], roi[2], shape[0] + 1)).astype(np.int)
                trunk0 = (pos0[1:] - pos0[:-1]).astype(np.int)
                pos1 = np.round(np.linspace(roi[1], roi[3], shape[1] + 1)).astype(np.int)
                trunk1 = (pos1[1:] - pos1[:-1]).astype(np.int)
                total = tf.slice(value, [0, roi[0], roi[1], 0], [1, roi[2] - roi[0], roi[3] - roi[1], x_shape[3]])
                output.append(
                    tf.concat([
                        tf.concat([
                            tf.nn.max_pool(block, (1, block_h, block_w, 1), (1, 1, 1, 1), "VALID")
                            for block_w, block in zip(trunk1, tf.split(line_block, trunk1, 2))
                        ], 2)
                        for block_h, line_block in zip(trunk0, tf.split(total, trunk0, 1))
                    ], 1)
                )
            batch_output.append(tf.concat(output, 0))
        return tf.stack(batch_output, 0)  # shape = (batch_size, sample_size, shape_h, shape_w, channel)

    @staticmethod
    def max_roi_align(x, batch_rois, shape):
        pass


class ModelBase:
    def __init__(self, config, name="base"):
        self.name = name
        self.config = config
        self.m_image_data = tf.placeholder(tf.uint8, [None, config.image_shape[0], config.image_shape[1], config.channel])
        self.m_label = tf.placeholder(tf.float32, [None, self.config.sample_size, 2])
        self.m_label_weight = tf.placeholder(tf.float32, [None, self.config.sample_size])
        self.m_proposal_regress = tf.placeholder(tf.float32, [None, self.config.sample_size, 4])
        self.m_proposal_regress_weight = tf.placeholder(tf.float32, [None, self.config.sample_size])
        self.predict_label = None
        self.predict_proposal_regress = None
        self.loss_label = None
        self.loss_proposal_regress = None
        self.loss = None

    def model_old(self):  # not use roi pooling
        h_00 = tl.batch_normalization(tf.cast(self.m_image_data, tf.float32))  # 256*256

        h_01 = tl.conv2d(h_00, 8, 3)  # 256*256
        h_02 = tl.conv2d(h_01, 8, 3)  # 256*256
        h_03 = tl.max_pooling2d(h_02)  # 128*128

        h_04 = tl.conv2d(h_03, 16, 3)  # 128*128
        h_05 = tl.conv2d(h_04, 16, 3)  # 128*128
        h_06 = tl.max_pooling2d(h_05)  # 64*64

        h_07 = tl.conv2d(h_06, 32, 3)  # 64*64
        h_08 = tl.conv2d(h_07, 32, 3)  # 64*64
        h_09 = tl.max_pooling2d(h_08)  # 32*32

        h_10 = tl.conv2d(h_09, 32, 3)  # 32*32
        h_11 = tl.conv2d(h_10, 32, 3)  # 32*32
        h_12 = tl.max_pooling2d(h_11)  # 16*16

        conv_predict_label = tl.conv2d(h_12, len(self.config.anchors) * 2, 1, activation_fn=None)
        conv_predict_proposal_regress = tl.conv2d(h_12, len(self.config.anchors) * 4, 1, activation_fn=None)

        predict_label_logits = tf.reshape(conv_predict_label, [-1, self.config.sample_size, 2])
        self.predict_label = tf.nn.softmax(predict_label_logits)
        self.predict_proposal_regress = tf.reshape(conv_predict_proposal_regress, [-1, self.config.sample_size, 4])

        self.loss_label = tf.losses.softmax_cross_entropy(self.m_label, predict_label_logits, weights=self.m_label_weight)
        self.loss_proposal_regress = tf.losses.mean_squared_error(self.m_proposal_regress, self.predict_proposal_regress, weights=self.m_proposal_regress_weight)

        self.loss = self.loss_label + self.loss_proposal_regress
        return self.predict_label, self.predict_proposal_regress, self.loss

    def model(self):    # use roi pooling
        h_00 = tl.batch_normalization(tf.cast(self.m_image_data, tf.float32))  # 256*256

        h_01 = tl.conv2d(h_00, 8, 3)  # 256*256
        h_02 = tl.conv2d(h_01, 8, 3)  # 256*256
        h_03 = tl.max_pooling2d(h_02)  # 128*128

        h_04 = tl.conv2d(h_03, 16, 3)  # 128*128
        h_05 = tl.conv2d(h_04, 16, 3)  # 128*128

        # roi pooling
        rois = self.config.bbox / 2  # 图像256*256->128*128
        roi_pooling_shape = (4, 4)
        assert (rois[:, :, 2] - rois[:, :, 0] >= roi_pooling_shape[0]).all() and (rois[:, :, 2] - rois[:, :, 0] >= roi_pooling_shape[1]).all()
        h_06 = SpecialLayer.max_roi_pooling(h_05, self.config.proposal, roi_pooling_shape)  # batch_size * sample_size * roi_pooling_shape_height * roi_pooling_shape_width * channel [batch_size * sample_size * 4 * 4 * 16]

        h_07 = tl.conv3d(h_06, 8, (1, 3, 3))  # batch_size * sample_size * 2 * 2 * 8
        h_08 = tf.reshape(h_07, [-1, self.config.sample_size * 2 * 2 * 8])

        predict_label_logits = tf.reshape(tl.dense(h_08, self.config.sample_size * 2), [-1, self.config.sample_size, 2])
        self.predict_label = tf.nn.softmax(predict_label_logits)
        self.predict_proposal_regress = tf.reshape(tl.dense(h_08, self.config.sample_size * 4), [-1, self.config.sample_size, 4])

        self.loss_label = tf.losses.softmax_cross_entropy(self.m_label, predict_label_logits, weights=self.m_label_weight)
        self.loss_proposal_regress = tf.losses.mean_squared_error(self.m_proposal_regress, self.predict_proposal_regress, weights=self.m_proposal_regress_weight)

        self.loss = self.loss_label + self.loss_proposal_regress
        return self.predict_label, self.predict_proposal_regress, self.loss

    def clean(self):
        if os.path.exists(self.name):
            shutil.rmtree(self.name)
        os.mkdir(self.name)
        os.mkdir("%s/log" % self.name)
        os.mkdir("%s/model" % self.name)
        os.mkdir("%s/sample" % self.name)

    def train(self, start_step=0, restore=False):
        m_predict_label, m_predict_proposal_regress, m_loss = self.model()
        tf.summary.scalar("loss", m_loss)
        merged_summary_op = tf.summary.merge_all()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=0.05, global_step=global_step, decay_steps=100, decay_rate=0.90)
        op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(m_loss)

        de = DataEngine(self.config, "data/image")
        de.load_info("data/info")
        dataset = de.read_tfrecords("data/train")
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)
        if restore:
            saver.restore(sess, os.path.join("%s/model" % self.name, "model.ckpt-%d" % start_step))
        else:
            self.clean()
        summary_writer = tf.summary.FileWriter("%s/log" % self.name, sess.graph)

        vis = Visualization(self.config, 5., "%s/sample" % self.name)

        step = start_step
        while step < 10000:
            batch_imdbs = sess.run(next_batch)
            _, label, label_weight, proposal_regress, proposal_regress_weight = de.calc_target(batch_imdbs["kind"], batch_imdbs["gtb"], batch_imdbs["ignore"], regress_scale=5.)
            if np.sum(proposal_regress_weight) < 0.5 or np.sum(label_weight) < 0.5:
                continue
            _, v_loss = sess.run([op, m_loss], feed_dict={
                global_step: step,
                self.m_image_data: batch_imdbs["image"][np.newaxis, ...],
                self.m_label: label[np.newaxis, ...],
                self.m_label_weight: label_weight[np.newaxis, ...],
                self.m_proposal_regress: proposal_regress[np.newaxis, ...],
                self.m_proposal_regress_weight: proposal_regress_weight[np.newaxis, ...]
            })
            if step % 10 == 0:
                print("step %6d: loss = %f" % (step, v_loss))
                summary_writer.add_summary(sess.run(merged_summary_op, feed_dict={
                    self.m_image_data: batch_imdbs["image"][np.newaxis, ...],
                    self.m_label: label[np.newaxis, ...],
                    self.m_label_weight: label_weight[np.newaxis, ...],
                    self.m_proposal_regress: proposal_regress[np.newaxis, ...],
                    self.m_proposal_regress_weight: proposal_regress_weight[np.newaxis, ...]
                }), step)
            if step % 100 == 0:
                v_predict_label, v_predict_proposal_regress = sess.run([m_predict_label, m_predict_proposal_regress], feed_dict={self.m_image_data: batch_imdbs["image"][np.newaxis, ...]})
                vis.visualization(step, batch_imdbs["image"][np.newaxis, ...], batch_marked_bbox=batch_imdbs["bbox"][np.newaxis, ...], batch_predict_label=v_predict_label, batch_predict_proposal_regress=v_predict_proposal_regress)
                saver.save(sess, "%s/model" % self.name, global_step=step)
            step += 1


class DataEngine(DataEngineBase):
    def __init__(self, config: Config, image_directory, keep_size=256):
        super().__init__(config, keep_size)
        self.image_directory = image_directory

    def load_origin_data(self):
        pass

    def filter_func(self, imdb: Imdb):
        pass


class Model(ModelBase):
    def __init__(self, name="base"):
        config = Config((256, 256), 3, (16, 16), [[10, 10], [20, 20], [30, 30]])
        super().__init__(config, name)


if __name__ == '__main__':
    Model().train()
