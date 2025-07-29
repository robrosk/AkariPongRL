import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_actions: int):
        super(NeuralNetwork, self).__init__()
        
        # 1) Conv1: in_channels=4 → out_channels=32, kernel=8×8, stride=4, ReLU
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation='relu',
            data_format='channels_last',  # input shape: (batch, height, width, channels)
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
        )
        
        # 2) Conv2: in_channels=32 → out_channels=64, kernel=4×4, stride=2, ReLU
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation='relu',
            data_format='channels_last',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
        )
        
        # 3) Conv3: in_channels=64 → out_channels=64, kernel=3×3, stride=1, ReLU
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            data_format='channels_last',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
        )
        
        # 4) Flatten layer (no learned parameters)
        self.flatten = tf.keras.layers.Flatten()
        
        # 5) Dense: 3136 → 512, ReLU
        self.dense = tf.keras.layers.Dense(
            units=512,
            activation='relu',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
        )
        
        # 6) Actor head (logits over actions, which represents the probability each action is taken based on the expected reward (training) and state)
        self.logits = tf.keras.layers.Dense(
            units=num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
        )
        
        # 7) Critic head (scalar value that represents the expected reward based on the state)
        self.value = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
        )

    def call(self, inputs):
        """
        Forward pass:
          inputs: float32 tensor, shape = (batch_size, 84, 84, 4),
                  with pixel values normalized to [0,1].
        Returns:
          logits: (batch_size, num_actions)
          value:  (batch_size, 1)
        """
        # (a) Conv layers
        x = self.conv1(inputs)   # → (batch_size, 20, 20, 32)
        x = self.conv2(x)        # → (batch_size, 9, 9, 64)
        x = self.conv3(x)        # → (batch_size, 7, 7, 64)

        # (b) Flatten and FC
        x = self.flatten(x)      # → (batch_size, 3136)
        h = self.dense(x)        # → (batch_size, 512)

        # (c) Actor & Critic heads
        logits = self.logits(h)  # → (batch_size, num_actions)
        value = self.value(h)    # → (batch_size, 1)

        return logits, value

# Example of running a dummy forward pass:
if __name__ == "__main__":
    print("Testing Neural Network...")
    model = NeuralNetwork(num_actions=6)

    # Pass a dummy input to build the model
    dummy_input = tf.random.uniform(shape=(1, 84, 84, 4), minval=0, maxval=1, dtype=tf.float32)
    logits_out, value_out = model(dummy_input)

    print("Model summary:")
    model.summary()

    print("Logits shape:", logits_out.shape)  # Expect: (1, 6)
    print("Value shape: ", value_out.shape)   # Expect: (1, 1)
