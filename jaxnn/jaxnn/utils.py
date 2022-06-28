import random
import jax
import torch
import torchvision

def random_key():
    return jax.random.PRNGKey(random.randint(1, 9999999999))


def draw_bbox(img, bbox, cls_map):
    img = torch.from_numpy(img * 255).to(torch.uint8)
    bbox = torch.from_numpy(bbox)
    bbox = torch.clone(bbox)
    bbox[:, 1::2] = bbox[:, 1::2] * img.shape[1]
    bbox[:, 2::2] = bbox[:, 2::2] * img.shape[0]
    bbox = bbox.to(torch.int32)
    filter = []
    for i in range(bbox.shape[0]):
        if bbox[i, 0] >= 0:
            filter.append(bbox[i])
    bbox = torch.stack(filter)
    labels = [i for i in bbox[:, 0] if i >= 0]
    labels = cls_map[labels]
    img = img.permute((2,0,1))
    return torchvision.utils.draw_bounding_boxes(img, bbox[:, 1:], colors='red', labels=labels).permute((1,2,0))
