import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns


############################################## load the test set and the model weight################################
test_File_Path = "testset_tetragonal_fromULBD.h5"
weights_PATH = "model_ULBD_tetragonal.weights.h5"
#####################################################################################################################

# define model
drop_rate = 0.2
drop_rate_2 = 0.4
initializer = tf.keras.initializers.GlorotUniform()

FCNsg_inputs = keras.Input(shape=(8192, 1))

c1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(FCNsg_inputs)
p1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c1)
p1 = layers.Dropout(rate=drop_rate)(p1)

c2 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p1)
p2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c2)
p2 = layers.Dropout(rate=drop_rate)(p2)
c3 = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p2)
p3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c3)
p3 = layers.Dropout(rate=drop_rate)(p3)
c4 = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p3)
p4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c4)
p4 = layers.Dropout(rate=drop_rate)(p4)
c5 = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p4)
p5 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c5)
p5 = layers.Dropout(rate=drop_rate)(p5)
c6 = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p5)
p6 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c6)
p6 = layers.Dropout(rate=drop_rate)(p6)
c7 = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p6)
p7 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c7)
p7 = layers.Dropout(rate=drop_rate)(p7)
c8 = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p7)
p8 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c8)
p8 = layers.Dropout(rate=drop_rate)(p8)
c9 = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                   kernel_initializer=initializer)(p8)
p9 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c9)
p9 = layers.Dropout(rate=drop_rate)(p9)
c10 = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                    kernel_initializer=initializer)(p9)
p10 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c10)
p10 = layers.Dropout(rate=drop_rate)(p10)
c11 = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                    kernel_initializer=initializer)(p10)
p11 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c11)
p11 = layers.Dropout(rate=drop_rate)(p11)
c12 = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(),
                    kernel_initializer=initializer)(p11)
p12 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c12)
p12 = layers.Dropout(rate=drop_rate_2)(p12)
c13 = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)(p12)
p13 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c13)
p13 = layers.Dropout(rate=drop_rate_2)(p13)

c14 = layers.Conv1D(filters=68, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)(p13)
f1 = layers.Flatten()(c14)

FCNsg_outputs = layers.Softmax()(f1)

FCN = tf.keras.Model(inputs=[FCNsg_inputs], outputs=[FCNsg_outputs], name="FCN")
opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
# opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
FCN.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=4, name='top_4_accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)
FCN.summary()

ICSD_test = h5py.File(test_File_Path, 'r')
stack_test = ICSD_test['data'][:, :]
data_test = stack_test[:, :-1]
x_test = data_test.reshape(-1, data_test.shape[1], 1)
y_test_sg = stack_test[:, -1] - 75
y_test_sg_onehot = tf.keras.utils.to_categorical(y_test_sg, num_classes=68)

FCN.load_weights(weights_PATH)

results = FCN.evaluate(x_test, y_test_sg_onehot)
test_loss = results[0]
test_accuracy = results[1]
test_top2_accuracy = results[2]
test_top3_accuracy = results[3]
test_top4_accuracy = results[4]
test_top5_accuracy = results[5]

print(f"Test Loss: {test_loss}")
print(f"Test Top-1 Accuracy: {test_accuracy}")
print(f"Test Top-2 Accuracy: {test_top2_accuracy}")
print(f"Test Top-3 Accuracy: {test_top3_accuracy}")
print(f"Test Top-4 Accuracy: {test_top4_accuracy}")
print(f"Test Top-5 Accuracy: {test_top5_accuracy}")

y_pred = FCN.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = tf.math.confusion_matrix(y_test_sg, y_pred_classes)

conf_matrix_sum = tf.reduce_sum(conf_matrix, axis=1, keepdims=True)
conf_matrix_percent = conf_matrix / conf_matrix_sum

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_percent, annot=False, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

