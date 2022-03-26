from tensorflow import keras


class VGG11(keras.Model):
    def __init__(self, outputs, name="vgg11", **kwargs) -> None:
        super(VGG11, self).__init__(name=name, **kwargs)
        filters = 64
        self.nets = []
        for n_conv in [1, 1, 2, 2, 2]:
            for _ in range(n_conv):
                self.layers.add(
                    keras.layers.Conv2D(filters, 3, padding="same", activation="relu")
                )
            self.layers.add(
                keras.layers.Maxpool2D(2, 2)
            )
            filter = filter << 1
        self.layers.add(keras.layers.Flatten())
        self.layers.add(
            keras.layers.Dense(512, activation='relu')
        )
        self.layers.add(
            keras.layers.Dense(256, activation='relu')
        )
        self.layers.add(
            keras.layers.Dense(outputs)
        )
            
