from argparse import ArgumentError
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve
import zipfile
from mnist import train_images, train_labels, test_images, test_labels
import os
from PIL import Image
from multiprocessing import Pool, Manager
import numpy as np
from xml.etree import ElementTree
from multiprocessing import Lock


DATA_SET_BASE_DIR = '/tmp/jaxnn_datasets/'

DATASETS = {
    'cat_and_dog': {
        'url':
        'https://storage.googleapis.com/kaggle-data-sets/23777/30378/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220623%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220623T031634Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=29c6467d49300db347b352e80eec5f53501c22a431e10d5a9d9e80d2f450e50471c3d4d2b5fc55c85d7d4643dd601df9c123b61eec2f0c962ccfd7a98acc2fb1ba741cea8a2f863004913cc7fbbf781d3deabe276a041479e5ed4936a9273483d926a6c4d433ea54de1be8d080da389975ab4f31e0f281430a55f7f49c30372faa7c343f60e7520d1eeba67aa3e05dd9fa39f279369c861b65eae0dcac6a4b71090e29b8315a69959ec01a2029f47a53f54a9242b787327d916785353934a0832daf1bef47996f9f3c13765d966f0ecdfed1767a7f62599b2fc0a000547e771f4c1c53b0bab4f63f25646cbc39cca442e2ae1f325697d04d6a8b4b73ef3a1725',
        'filename': 'cat-and-dog.zip'
    },
    'face_mask': {
        'url':
        'https://storage.googleapis.com/kaggle-data-sets/667889/1176415/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220628%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220628T015358Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=a75db7d96c946382db313c8c90054fe0c592e6b0f78645172f5e8d04052404d15cc440785b7ae22f0166f2dc2bc73814cf394438b05bbf855ab32036ee90739f97411149823a37ce94a20ec4b8389b7a0de236fb041ef5dd22b0bf0856e472293cdd8239c2e24842579464df8022a29b25417c41bc4340949e51466604574068e405c096e68d6a784a0a61e6b92655b000ffe343737775b4d825071827a4ac50551e8add0a77851f3b5e8ff91da8a07ede22f4b64f6327f66d90934c4a446be7ef7925f5c9c17e39f11173afaa10cbd379ad52f4d7dadb428bdd05c39b0b80b93d27ddfb15e2ff8010cec96338b41779a1a3b086bbf8fb5c49fd03ff6334f5db',
        'filename': 'face-mask.zip'
    }
}

class ClassMap:
    
    def __init__(self, *args) -> None:
        self.tags = ['background']
        self.tags += args
        self.tags_map = {tag: idx for idx, tag in enumerate(self.tags)}
    
    def __getitem__(self, keys):
        if hasattr(keys, '__next__') or isinstance(keys, (tuple, list)):
            return [self.__getitem__(k) for k in keys]
        if isinstance(keys, str):
            return self.tags_map.get(keys)
        # elif isinstance(keys, int):
        return self.tags[keys]

def load_img(pack):
    (container, _index, resizer) = pack
    img = container[_index]
    if isinstance(img, str):
        img = Image.open(img).resize(resizer)
        img = np.array(img)
    container[_index] = img
    return img


def _yield(obj):
    while True:
        yield obj


def _path_reveal(filename):
    if not os.path.exists(DATA_SET_BASE_DIR):
        os.makedirs(DATA_SET_BASE_DIR)
    return DATA_SET_BASE_DIR + filename


def mnist_dataloader(seed=None,
                     batch_size=50,
                     repeat=True,
                     valid_ratio=.2,
                     map_fn=None,
                     *args,
                     **kwargs):
    return _dataloader(x=train_images(),
                       y=train_labels(),
                       batch_size=batch_size,
                       seed=seed,
                       repeat=repeat,
                       valid_ratio=valid_ratio,
                       map_fn=map_fn)


def _dataloader(x, y, batch_size, repeat, valid_ratio, map_fn, seed):
    valid_len = int(valid_ratio * x.shape[0])
    indicates = np.arange(len(x))
    if seed:
        np.random.seed(seed)
    np.random.shuffle(indicates)
    x = x[indicates]
    y = y[indicates]
    valid_x = x[:valid_len]
    valid_y = y[:valid_len]
    while True:
        if map_fn:
            valid_x = map_fn(valid_x)
        yield _split(x[valid_len:], y[valid_len:], batch_size,
                     map_fn), (valid_x, valid_y)
        if not repeat:
            break


def mnist_testdata():
    return test_images(), test_labels()


_to_mb = lambda b: b / (1024 * 1024)


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


def cat_and_dog_dataloader(
    _type,
    resize=(224, 224),
    seed=None,
    batch_size=50,
    repeat=True,
    valid_ratio=.2,
    map_fn=None,
):
    _POOL = Pool()
    _MGR = Manager()
    dataset = DATASETS['cat_and_dog']
    file_dir = _get_file_dir(remote_url=dataset['url'],
                             filename=dataset['filename'])

    def _cat_all_img(path_suffix, coll):
        img_path = file_dir + path_suffix
        for img in os.listdir(img_path):
            if img.endswith('.jpg'):
                coll.append(img_path + os.sep + img)
        return coll

    if _type == 'train':
        cats = _cat_all_img('/training_set/training_set/cats', [])
        dogs = _cat_all_img('/training_set/training_set/dogs', [])
    elif _type == 'test':
        valid_ratio = 0
        cats = _cat_all_img('/test_set/test_set/cats', [])
        dogs = _cat_all_img('/test_set/test_set/dogs', [])
    else:
        raise ArgumentError(_type, "_type must be either 'tarin' or 'test'")
    y = np.concatenate(
        (np.full(shape=(len(cats), ),
                 fill_value=0), np.full(shape=(len(dogs), ), fill_value=1)))
    x = np.arange(len(y))
    train = _MGR.list()
    train += cats
    train += dogs
    del cats, dogs

    def _lazy_map(_x_index):
        xs = _POOL.map(load_img, zip(_yield(train), _x_index, _yield(resize)))
        _x = np.stack(xs)
        if map_fn:
            _x = map_fn(_x)
        return _x

    return _dataloader(x=x,
                       y=y,
                       batch_size=batch_size,
                       seed=seed,
                       repeat=repeat,
                       valid_ratio=valid_ratio,
                       map_fn=_lazy_map)


def _split(x, y, batch_size, map_fn):
    all = len(x)
    n_batch = all // batch_size
    for i in range(n_batch):
        left = batch_size * i
        right = left + batch_size
        if right <= all:
            batch_x, batch_y = x[left:right], y[left:right]
            if map_fn:
                batch_x = map_fn(batch_x)
            yield batch_x, batch_y

def _scale_bbox(bbox, ori, resize):
    ori_h, ori_w = ori
    rsz_h, rsz_w = resize
    h_scale = rsz_h / ori_h
    w_scale = rsz_w / ori_w
    bbox[:, 0] = np.round(bbox[:, 0] * w_scale)
    bbox[:, 1] = np.round(bbox[:, 1] * h_scale)
    bbox[:, 2] = np.round(bbox[:, 2] * w_scale)
    bbox[:, 3] = np.round(bbox[:, 3] * h_scale)
    return bbox


def face_mask_dataloader(
    resize=(256, 256),
    seed=None,
    batch_size=50,
    repeat=True,
    valid_ratio=.2,
    map_fn=None,
):

    # _POOL = Pool()
    # _MGR = Manager()
    dataset = DATASETS['face_mask']
    file_dir = _get_file_dir(remote_url=dataset['url'],
                             filename=dataset['filename'])
    cls_map = ClassMap('with_mask', 'without_mask', 'mask_weared_incorrect')
    annos = file_dir + os.sep + 'annotations' + os.sep

    def _parse_anno(xf):
        root = ElementTree.parse(annos + xf).getroot()
        anno = dict()
        fname = root.find('filename').text
        fsize = root.find('size')
        ori_h = int(fsize.find('height').text)
        ori_w = int(fsize.find('width').text)
        img = Image.open(file_dir + os.sep + 'images' + os.sep +
                         fname).resize(resize).convert('RGB')
        img = np.array(img, dtype='float32') / 255.
        label = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            l = [
                cls_map[obj.find('name').text],
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)
            ]
            label.append(np.array(l, dtype='float32'))
        label = np.stack(label)
        label[:, 1:] = _scale_bbox(label[:, 1:], (ori_h, ori_w), resize)
        label[:, 1::2] = label[:, 1::2] / resize[0]
        label[:, 2::2] = label[:, 2::2] / resize[1]
        return img, label
    imgs, labels = [], []
    max_n_label = [0]
    executor = ThreadPoolExecutor()
    locker = Lock()
    def _execute(xf, img_col, label_col, lock):
        try:
            img, label = _parse_anno(xf)
            try:
                lock.acquire()
                max_n_label[0] = max(label.shape[0], max_n_label[0])
                img_col.append(img)
                label_col.append(label)
            finally:
                lock.release()
        except Exception as e:
            print(e)
    for xf in os.listdir(annos):
        executor.submit(_execute, xf, imgs, labels, locker)
        # _execute(xf, imgs, labels, locker)
    executor.shutdown()
    # fill labels
    for i, label in enumerate(labels):
        fill_require = max_n_label[0] - label.shape[0]
        if fill_require > 0:
            fill = np.full((fill_require, label.shape[-1]), fill_value=-1., dtype='float32')
            labels[i] = np.concatenate((label, fill))
    imgs, labels = np.stack(imgs), np.stack(labels)
    return _dataloader(x=imgs,
                       y=labels,
                       batch_size=batch_size,
                       repeat=repeat,
                       valid_ratio=valid_ratio,
                       map_fn=map_fn,
                       seed=seed), cls_map


if __name__ == '__main__':
    dataloader, cls_map = face_mask_dataloader()
    (train_iter, (x, y)) = next(dataloader)
    print(x.shape, y.shape)
