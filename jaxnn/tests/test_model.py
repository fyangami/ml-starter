import jaxnn
from unittest import TestCase


class TestModel(TestCase):
    
    def test01(self):
        model = jaxnn.Model([])
        print(model)
        assert 1 == 1

# def test01():
#     model = jaxnn.Model([])
#     print(model)
#     assert 1 == 1
