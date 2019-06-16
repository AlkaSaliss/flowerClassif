import tensorflow as tf


simple_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 5, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal()),
    tf.keras.layers.Conv2D(64, 5, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal()),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 5, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal()),
    tf.keras.layers.Conv2D(128, 5, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal()),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu',
                          kernel_initializer=tf.keras.initializers.he_normal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(5, activation='softmax')
])

simple_cnn.compile(
    tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['acc']
)

print(simple_cnn.summary())


