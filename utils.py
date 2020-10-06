import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
# from IPython.display import display
from seaborn import color_palette

from config import cfg


# 获取类别
def read_class_names(class_file_name):
    """
    :param class_file_name: class 文件路径
    :return:
    """
    names = {}
    with open(class_file_name, 'r') as data:
        # 获取类名和下标，用于数值和类之间的转换
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    print(names)
    return names


# 获取 anchor
def get_anchors(anchors_path):
    """
    :param anchors_path: anchor 文件路径
    :return:
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    # anchors = anchors.reshape(-1, 2)
    anchors = anchors.reshape((3, 3, 2))
    # print("anchors.reshape:{}".format(anchors))
    return anchors


# 读取 image data 的路径
def read_data_path(file_path):
    image_path_list = []
    with open(file_path) as file:
        line_list = file.readlines()
        for line_info in line_list:
            data_info = line_info.strip()
            image_path = data_info.split()[0]
            image_path_list.append(image_path)
            pass
        pass

    return image_path_list
    pass


# 读取数据
def read_data_line(file_path):
    data_line_list = []
    with open(file_path) as file:
        line_list = file.readlines()
        for line_info in line_list:
            data_info = line_info.strip().split()
            data_line_list.append(data_info)

    return data_line_list


# 打框
def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Draws detected boxes.

    Args:
        img_names: A list of input images names.
        boxes_dict: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size: The input size of the model.

    Returns:
        None.
    """
    colors = ((np.array(color_palette("hls", 2)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
                                         boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font='./futura/futur.ttf',
                                  size=(img.size[0] + img.size[1]) // 100)
        resize_factor = \
            (img.size[0] / model_size[0], img.size[1] / model_size[1])
        for cls in range(len(class_names)):
            print("cls: ", cls)
            print("boxes_dict in utils: ", boxes_dict)
            print("boxes_dict length: ", len(boxes_dict))
            for i in range(3):
                boxes = boxes_dict[i][cls]
                # print("boxes: ", boxes)
                if np.size(boxes) != 0:
                    color = colors[cls]
                    for box in boxes:
                        print("box: ", box)
                        xy, confidence = box[:4], box[4]
                        xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                        x0, y0 = xy[0], xy[1]
                        thickness = (img.size[0] + img.size[1]) // 200
                        for t in np.linspace(0, 1, thickness):
                            xy[0], xy[1] = xy[0] + t, xy[1] + t
                            xy[2], xy[3] = xy[2] - t, xy[3] - t
                            draw.rectangle(xy, outline=tuple(color))
                        text = '{} {:.1f}%'.format(class_names[cls],
                                                   confidence * 100)
                        text_size = draw.textsize(text, font=font)
                        draw.rectangle(
                            [x0, y0 - text_size[1], x0 + text_size[0], y0],
                            fill=tuple(color))
                        draw.text((x0, y0 - text_size[1]), text, fill='black',
                                  font=font)

        # display(img)
        img.save('test.jpg')
        print('image saved!')


# (x, y, w, h) --> (xmin, ymin, xmax, ymax)
def bbox_xywh_dxdy(boxes):
    boxes = np.array(boxes)
    boxes_dxdy = np.concatenate([boxes[..., :2] - boxes[..., 2:] * 0.5,
                                 boxes[..., :2] + boxes[..., 2:] * 0.5], axis=-1)

    return boxes_dxdy
    pass


# (xmin, ymin, xmax, ymax) --> (x, y, w, h)
def bbox_dxdy_xywh(boxes):
    boxes = np.array(boxes)
    # 转换 [xmin, ymin, xmax, ymax] --> [x, y, w, h] bounding boxes 结构
    bbox_xywh = np.concatenate([(boxes[2:] + boxes[:2]) * 0.5,
                                boxes[2:] - boxes[:2]], axis=-1)

    return bbox_xywh
    pass


# IOU
def bbox_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # 计算 面积
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 交集的 左上角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    # 交集的 右下角坐标
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算交集矩形框的 high 和 width
    inter_section = np.maximum(right_down - left_up, 0.0)

    # 两个矩形框的 交集 面积
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # 两个矩形框的并集面积
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算 iou
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


# 非极大值抑制
def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    :param iou_threshold: iou 阈值
    :param sigma:
    :param method: 方法
    :return:
    """
    # 获取 bbox 中类别种类的 list
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        # 构建一个 bbox batch size 大小的 list，
        # 类别与 cls 相同的为 list内容为 True，不同的额外 False
        # 比如 bboxes = [[12, 12, 94, 94, 0.78, 0],
        #                [34, 34, 64, 64, 0.88, 1],
        #                [78, 78, 124, 124, 0.98, 0],
        #                [52, 52, 74, 74, 0.78, 1]
        #               ]
        # 第一次遍历得到 [True, False, True, False] 这样的 cls_mask list
        # 第二次遍历得到 [False, True, False, True] 这样的 cls_mask list
        cls_mask = (bboxes[:, 5] == cls)
        # 第一次遍历得到 [[12, 12, 94, 94, 0.78, 0], [78, 78, 124, 124, 0.98, 0]] 这样的 cls_bboxes list
        # 第二次遍历得到 [[34, 34, 64, 64, 0.88, 1], [52, 52, 74, 74, 0.78, 1]] 这样的 cls_bboxes list
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            # 获取最大概率值的下标
            max_ind = np.argmax(cls_bboxes[:, 4])
            # 概率值最大的  bbox 为最佳 bbox
            best_bbox = cls_bboxes[max_ind]
            # 将所有 最好的 bbox 放到一个 list 中
            best_bboxes.append(best_bbox)
            # 将概率最大的那个 bbox 移除后 剩下的 bboxes
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # 计算 best bbox 与剩下的 bbox 之间的 iou
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            # 构建一个 长度为 len(iou) 的 list，并填充 1 值
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                # 将大于阈值的 iou，其对应 list 的值设置为 0，用于下面对该值进行移除
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            # 移除 大于阈值 的 bboxes，如此重复，直至 cls_bboxes 为空
            # 将大于阈值的 bbox 概率设置为 零值
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            # 保留概率 大于 零值 的 bbox
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


# 处理后的盒子
def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    """
    :param pred_bbox: 预测的 bbox
    :param org_img_shape: 原始图像的 shape
    :param input_size: 输入的大小
    :param score_threshold: 得分阈值
    :return:
    """
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    # bbox 坐标
    pred_xywh = pred_bbox[:, 0:4]
    # bbox 置信度
    pred_conf = pred_bbox[:, 4]
    # bbox 概率
    pred_prob = pred_bbox[:, 5:]

    # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    # 将预测的 x 的坐标(xmin, xmax) pred_coor[:, 0::2] 减去空白区域 dw 后，
    # 除以缩放比率，得到原图 x 方向的大小
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    # 将预测的 y 的坐标(ymin, ymax) pred_coor[:, 1::2] 减去空白区域 dh 后，
    # 除以缩放比率，得到原图 y 方向的大小
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (3) clip some boxes those are out of range 处理那些超出原图大小范围的 bboxes
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)

    # 处理不正常的 bbox
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4) discard some invalid boxes 丢弃无效的 bbox
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    # np 的 逻辑 and
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5) discard some boxes with low scores 丢弃分值过低的 bbox
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def get_anchors():
    anchors = "1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875"
    anchors = np.array(anchors.split(', '), dtype=np.float32)
    return anchors.reshape((3, 3, 2))


def image_preporcess(image, target_size, gt_boxes=None):
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ih, iw = target_size
    h, w, _ = image.shape

    # print("ih, iw:", ih, iw)
    # print("h,  w: ", h,  w )

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def build_boxes(inputs):
    """Computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes
