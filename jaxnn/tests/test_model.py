import unittest
import jaxnn
from unittest import TestCase

from jaxnn.model import Model
from jaxnn import nn
from jaxnn import optimizers
from jaxnn import loss
import jax
from jax import numpy as jnp

class TestModel(TestCase):
    
    # def test01(self):
    #     dataloader = jaxnn.dataset.mnist_dataloader('train', batch_size=256)
    #     model = Model([
    #         nn.conv2d(n_filter=32, kernel_size=(3,3), strides=1),
    #         nn.relu(),
    #         # nn.conv2d(n_filter=32, kernel_size=(3,3), strides=1),
    #         # nn.relu(),
    #         nn.flatten(),
    #         # nn.dense(256),
    #         # nn.relu(),
    #         nn.dropout(),
    #         nn.dense(10),
    #         nn.softmax()
    #     ])
    #     model.fit(data_loader=dataloader,
    #             optimizer=optimizers.sgd(lr=1e-1),
    #             loss_fn=loss.categorical_cross_entropy(),
    #             epochs=100,
    #             metrics=['accuracy'])
    
    def test02(self):
        dataloader, cls_map = jaxnn.dataset.mnist_dataloader('train', batch_size=50)
        model = Model([
            nn.conv2d(n_filter=64, kernel_size=3, strides=1, padding='SAME'),
            nn.relu(),
            nn.maxpool2d(window=2, strides=2, padding='VALID'),
            nn.conv2d(n_filter=128, kernel_size=3, strides=1, padding='SAME'),
            nn.relu(),
            nn.maxpool2d(window=2, strides=2, padding='VALID'),
            nn.conv2d(n_filter=256, kernel_size=3, strides=1, padding='SAME'),
            nn.relu(),
            nn.maxpool2d(window=2, strides=2, padding='VALID'),
            nn.flatten(),
            nn.dense(256),
            nn.dense(len(cls_map)),
            nn.softmax()
        ])
        model.fit(dataloader,
                optimizer=optimizers.sgd(1e-3),
                loss_fn=loss.categorical_cross_entropy(from_logits=False),
                epochs=50,
                metrics=['accuracy'])


if __name__ == '__main__':
    unittest.main()
