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
    
    def test01(self):
        dataloader = jaxnn.dataset.mnist_dataloader(map_fn=lambda x: x / 255., batch_size=256)
        model = Model([
            nn.conv2d(n_filter=32, kernel_size=(3,3), strides=1),
            nn.relu(),
            # nn.conv2d(n_filter=32, kernel_size=(3,3), strides=1),
            # nn.relu(),
            nn.flatten(),
            # nn.dense(256),
            # nn.relu(),
            nn.dropout(),
            nn.dense(10),
            nn.softmax()
        ])
        model.fit(data_loader=dataloader,
                optimizer=optimizers.sgd(lr=1e-1),
                loss_fn=loss.categorical_cross_entropy(),
                epochs=100,
                metrics=['accuracy'])

# def test01():
#     model = jaxnn.Model([])
#     print(model)
#     assert 1 == 1

if __name__ == '__main__':
    unittest.main()
