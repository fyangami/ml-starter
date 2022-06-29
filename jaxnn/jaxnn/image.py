import numpy as np
from jax import numpy as jnp
import torch
import torchvision

def draw_bbox(img, bbox, cls_map=None):
    if isinstance(bbox, jnp.DeviceArray):
        bbox = np.array(bbox)
    if isinstance(img, jnp.DeviceArray):
        img = np.array
    img = torch.from_numpy(img * 255).to(torch.uint8)
    bbox = torch.from_numpy(bbox)
    bbox = torch.clone(bbox)
    _, n_col = bbox.shape
    pos_start = 0 if n_col == 4 else 1
    bbox[:, pos_start::2] = bbox[:, pos_start::2] * img.shape[1]
    bbox[:, pos_start+1::2] = bbox[:, pos_start+1::2] * img.shape[0]
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
    img = img.permute((2,0,1))
    return torchvision.utils.draw_bounding_boxes(img, bbox[:, pos_start:], colors='red', labels=labels).permute((1,2,0))


def multibox_prior(fmap, scales, ratios):
    # data.shape (b, h, w, c)
    # generate `bbox` for per pixel.
    pair = [(scales[0], ratio) for ratio in ratios]
    pair += [(scale, ratios[0]) for scale in scales[1:]]
    (_, h, w, _) = fmap.shape
    # shift per pixel to center
    offset = .5
    pixel_x, pixel_y = jnp.meshgrid(jnp.arange(w, dtype='float32'), jnp.arange(h, dtype='float32'))
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
    a_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    b_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    inter_xy_min = jnp.maximum(anchors[:, None, :2], bboxes[:, :2])
    inter_xy_max = jnp.minimum(anchors[:, None, 2:], bboxes[:, 2:])
    inters = jnp.clip(inter_xy_max - inter_xy_min, 0)
    inter_area = inters[:, :, 0] * inters[:, :, 1]
    union_area = a_area[:, None] + b_area - inter_area
    return inter_area / union_area