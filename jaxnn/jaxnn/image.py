import numpy as np
from jax import numpy as jnp
import torch
import torchvision


def draw_bbox(img, bbox, cls_map=None):
    if isinstance(bbox, jnp.DeviceArray):
        bbox = np.array(bbox)
    if isinstance(img, jnp.DeviceArray):
        img = np.array(img)
    img = torch.from_numpy(img * 255).to(torch.uint8)
    bbox = torch.from_numpy(bbox)
    bbox = torch.clone(bbox)
    _, n_col = bbox.shape
    pos_start = 0 if n_col == 4 else 1
    bbox[:, pos_start::2] = bbox[:, pos_start::2] * img.shape[1]
    bbox[:, pos_start + 1::2] = bbox[:, pos_start + 1::2] * img.shape[0]
    bbox = bbox.to(torch.int32)
    labels = None
    if pos_start == 1:
        filter = []
        for i in range(bbox.shape[0]):
            if bbox[i, 0] >= 0:
                filter.append(bbox[i])
        bbox = torch.stack(filter)
        labels = [i for i in bbox[:, 0] if i >= 0]
        labels = cls_map[labels]
    img = img.permute((2, 0, 1))
    return torchvision.utils.draw_bounding_boxes(img,
                                                 bbox[:, pos_start:],
                                                 colors='red',
                                                 labels=labels).permute(
                                                     (1, 2, 0))


def multibox_prior(fmap, scales, ratios):
    # data.shape (b, h, w, c)
    # generate `bbox` for per pixel.
    pair = [(scales[0], ratio) for ratio in ratios]
    pair += [(scale, ratios[0]) for scale in scales[1:]]
    (_, h, w, _) = fmap.shape
    # shift per pixel to center
    offset = .5
    pixel_x, pixel_y = jnp.meshgrid(jnp.arange(w, dtype='float32'),
                                    jnp.arange(h, dtype='float32'))
    pixel_x, pixel_y = (pixel_x + offset) / w, (pixel_y + offset) / h

    all = []
    for (scale, ratio) in pair:
        # w * s * sqrt(r)
        half_bw = w * scale * jnp.sqrt(ratio) / 2
        # h * s / sqrt(r)
        half_bh = h * scale / jnp.sqrt(ratio) / 2
        xmin = pixel_x - half_bw
        ymin = pixel_y - half_bh
        xmax = pixel_x + half_bw
        ymax = pixel_y + half_bh
        box = jnp.stack((xmin, ymin, xmax, ymax), axis=-1)
        all.append(box)
    all = jnp.stack(all)
    return jnp.reshape(all, (-1, 4))


def iou(anchors, bboxes):
    # xmin ymin xmax ymax
    a_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    b_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    inter_xy_min = jnp.maximum(anchors[:, None, :2], bboxes[:, :2])
    inter_xy_max = jnp.minimum(anchors[:, None, 2:], bboxes[:, 2:])
    inters = jnp.clip(inter_xy_max - inter_xy_min, 0)
    inter_area = inters[:, :, 0] * inters[:, :, 1]
    union_area = a_area[:, None] + b_area - inter_area
    return inter_area / union_area


def assign_anchor_to_bbox(anchors, bboxes, threshold=.5):
    n_anchors, n_bboxes = anchors.shape[0], bboxes.shape[0]
    bbox_iou = iou(anchors=anchors, bboxes=bboxes)
    # iou = np.array(iou)
    col_discard = jnp.full(fill_value=-1, dtype='float32', shape=(n_bboxes))
    row_discard = jnp.full(fill_value=-1, dtype='float32', shape=(n_anchors))

    anchor_bbox_map = jnp.full((n_anchors, ), fill_value=-1)

    for _ in range(n_bboxes):
        maximum_row = jnp.argmax(bbox_iou, axis=0)
        anc_idx = np.max(maximum_row)
        maximum_col = jnp.argmax(bbox_iou[anc_idx, :])
        bbox_idx = np.max(maximum_col)
        if bbox_iou[anc_idx, bbox_idx] < threshold:
            break
        anchor_bbox_map = anchor_bbox_map.at[anc_idx].set(bbox_idx)
        bbox_iou = bbox_iou.at[anc_idx, :].set(col_discard)
        bbox_iou = bbox_iou.at[:, bbox_idx].set(row_discard)
    return jnp.array(anchor_bbox_map)


def corner_to_center(box):
    cw, ch = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    return jnp.stack((cw, ch, w, h), axis=-1)


def center_to_corner(box):
    half_w, half_h = box[:, 2] / 2., box[:, 3] / 2.
    return jnp.stack((box[:, 0] - half_w, box[:, 0] + half_w,
                      box[:, 1] - half_h, box[:, 1] + half_h),
                     axis=-1)

def offset_boxes(anchors, bboxes, eps=1e-6):
    c_anc = corner_to_center(anchors)
    c_bbox = corner_to_center(bboxes)
    offset_xy = 10 * (c_bbox[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_hw = 5 * jnp.log(eps + c_bbox[:, 2:] / c_anc[:, 2:])
    return jnp.concatenate((offset_xy, offset_hw), axis=-1)

def offset_inverse(anchors, offsets):
    c_anc = corner_to_center(anchors)
    pred_xy = (offsets[:, :2] * c_anc[:, 2:] / 10) + c_anc[:, :2]
    pred_wh = jnp.exp(offsets[:, 2:] / 5) * c_anc[:, 2:]
    pred = jnp.concatenate((pred_xy, pred_wh), axis=-1)
    return center_to_corner(pred)

def multibox_target(anchors, bboxes):
    offsets, masks, labels = [], [], []
    for i in range(len(bboxes)):
        label = bboxes[i, :, :]
        anchor_bbox_map = assign_anchor_to_bbox(anchors=anchors, bboxes=label[:, 1:])
        indicates, = jnp.nonzero(anchor_bbox_map >= 0)
        if len(indicates) == 0:
            continue
        _mask = jnp.repeat(jnp.expand_dims(anchor_bbox_map >= 0, axis=-1), 4, axis=-1).astype('float32')
        bbox_indicates = anchor_bbox_map[np.array(indicates)]
        _bbox = jnp.zeros((anchors.shape[0], 4), dtype='float32')
        _class = jnp.full((anchors.shape[0], 1), 0, dtype='int32')
        # bbox_iou = bbox_iou.at[anc_idx, :].set(col_discard)
        _class = _class.at[indicates, :].set(label[bbox_indicates, :1] + 1)
        _bbox = _bbox.at[indicates, :].set(label[bbox_indicates, 1:])
        _offset = offset_boxes(_bbox, anchors) * _mask
        offsets.append(_offset)
        labels.append(_class)
        masks.append(_mask)
    return jnp.stack(offsets), jnp.stack(masks), jnp.stack(labels)
