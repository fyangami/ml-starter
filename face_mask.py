from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
import os
from PIL import Image
from urllib.request import urlretrieve
from xml.etree import ElementTree
import zipfile
import numpy
import torch
from torch.utils.data import DataLoader, Dataset

DATA_SET_BASE_DIR = os.environ['HOME'] + '/.jaxnn_datasets/'

DATASETS = {
    'cat_and_dog': {
        'url':
        'https://storage.googleapis.com/kaggle-data-sets/23777/30378/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220629%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220629T071213Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=90840093a05d6110febbe46c6b3891152f4547a69bff8d0b69a232ce945dd1292850e12a8ecdbabb4ac2c892ab3482c4496387a2b09d987b70c872df265497fea7cb2950d773908312eb6cf2dee44772b3351587bc423694293c6605206a63ebd14680f6b77d3e86fa4c574e279cb5b59bfec6a5a4f0abb22f00df8e8c57d74755349254edac43a5a06349c265083f98c26874fb2f8377ae286833a2fad594bb72a5153c7b401993d288fd8688769f2bd51247f8d1b67da2d48d7b26e50ac6f0830854e6f432c5d4b5a7e055ca10aca59c44b7dfbc5bc024fa4524e23e6f0a56ff89001a7bbf557eec1f9403050d6ffefba62213a5d62d7ecae0dcd0245d9287',
        'filename': 'cat-and-dog.zip'
    },
    'face_mask': {
        'url':
        'https://storage.googleapis.com/kaggle-data-sets/667889/1176415/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220628%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220628T015358Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=a75db7d96c946382db313c8c90054fe0c592e6b0f78645172f5e8d04052404d15cc440785b7ae22f0166f2dc2bc73814cf394438b05bbf855ab32036ee90739f97411149823a37ce94a20ec4b8389b7a0de236fb041ef5dd22b0bf0856e472293cdd8239c2e24842579464df8022a29b25417c41bc4340949e51466604574068e405c096e68d6a784a0a61e6b92655b000ffe343737775b4d825071827a4ac50551e8add0a77851f3b5e8ff91da8a07ede22f4b64f6327f66d90934c4a446be7ef7925f5c9c17e39f11173afaa10cbd379ad52f4d7dadb428bdd05c39b0b80b93d27ddfb15e2ff8010cec96338b41779a1a3b086bbf8fb5c49fd03ff6334f5db',
        'filename': 'face-mask.zip'
    }
}

class _Dataset(Dataset):
    
    def __init__(self, imgs, targets) -> None:
        self.imgs = imgs
        self.targets = targets
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]

class _ClassMap:
    def __init__(self, *args, fill_none=True) -> None:
        self.tags = []
        if fill_none:
            self.tags.append('none')
        self.tags += args
        self.tags_map = {tag: idx for idx, tag in enumerate(self.tags)}

    def __getitem__(self, keys):
        if hasattr(keys, '__next__') or isinstance(keys, (tuple, list)):
            return [self.__getitem__(k) for k in keys]
        if isinstance(keys, str):
            return self.tags_map.get(keys)
        # elif isinstance(keys, int):
        return self.tags[keys]
    
    def __len__(self):
        return len(self.tags)


_to_mb = lambda b: b / (1024 * 1024)

def _path_reveal(filename):
    if not os.path.exists(DATA_SET_BASE_DIR):
        os.makedirs(DATA_SET_BASE_DIR)
    return DATA_SET_BASE_DIR + filename

def _get_file_dir(remote_url, filename):
    filename = _path_reveal(filename)
    if not os.path.exists(filename):

        def _progress_log(blk_num, blk_size, total_size):
            total_size = _to_mb(total_size)
            cur = _to_mb(blk_num * blk_size)
            print(f'\rdownloading: {cur:.2f}Mb\\{total_size:.2f}Mb', end='')

        print('start download remote file.')
        urlretrieve(url=remote_url,
                    filename=filename,
                    reporthook=_progress_log)
        print('\ndownload done.')
    zip_filename = filename
    dir = filename.strip('.zip')
    if not os.path.exists(dir):
        zip_file = zipfile.ZipFile(zip_filename)
        os.makedirs(dir)
        print('start extract file.')
        zip_file.extractall(dir)
        print('extract file done.')
    return dir


def _scale_bbox(bbox, ori, resize):
    ori_h, ori_w = ori
    rsz_h, rsz_w = resize
    h_scale = rsz_h / ori_h
    w_scale = rsz_w / ori_w
    bbox[:, 0] = torch.round(bbox[:, 0] * w_scale)
    bbox[:, 1] = torch.round(bbox[:, 1] * h_scale)
    bbox[:, 2] = torch.round(bbox[:, 2] * w_scale)
    bbox[:, 3] = torch.round(bbox[:, 3] * h_scale)
    return bbox

def face_mask_dataloader(resize=(256, 256),
                         batch_size=50):
    dataset = DATASETS['face_mask']
    file_dir = _get_file_dir(remote_url=dataset['url'],
                             filename=dataset['filename'])
    annos = file_dir + os.sep + 'annotations' + os.sep
    cls_map = _ClassMap('with_mask', 'without_mask', 'mask_weared_incorrect')
    def _parse_anno(xf):
        root = ElementTree.parse(annos + xf).getroot()
        fname = root.find('filename').text
        fsize = root.find('size')
        ori_h = int(fsize.find('height').text)
        ori_w = int(fsize.find('width').text)
        img = Image.open(file_dir + os.sep + 'images' + os.sep +
                         fname).resize(resize).convert('RGB')
        img = numpy.array(img, dtype='float32') / 255.
        img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))
        _labels, _bboxes = [], []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            _labels.append(cls_map[obj.find('name').text])
            box = [
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)
            ]
            _bboxes.append(box)
        _bboxes = torch.tensor(_bboxes, dtype=torch.float32)
        _bboxes = _scale_bbox(_bboxes, (ori_h, ori_w), resize)
        _labels = torch.tensor(_labels, dtype=torch.int32)
        return img, _bboxes, _labels

    imgs, targets = [], []
    executor = ThreadPoolExecutor()
    locker = Lock()

    def _execute(xf):
        try:
            img, img_bboxes, img_labels = _parse_anno(xf)
            target = {
                'boxes': img_bboxes,
                'labels': img_labels
            }
            try:
                locker.acquire()
                imgs.append(img)
                targets.append(target)
            finally:
                locker.release()
        except Exception as e:
            print(e)

    for xf in os.listdir(annos):
        executor.submit(_execute, xf)
        # _execute(xf)
    executor.shutdown()
    def _collate_fn(batch):
        x, y = [], []
        for img, target in batch:
            x.append(img)
            y.append(target)
        return torch.stack(x, dim=0), y
    return iter(DataLoader(_Dataset(imgs=imgs, targets=targets), shuffle=True, batch_size=batch_size, collate_fn=_collate_fn))


if __name__ == '__main__':
    data = face_mask_dataloader(batch_size=10)
    x, y = next(data)
    print(x)
