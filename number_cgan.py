#type: ignore
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import Dependencies
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Reshape, Embedding, Multiply, Concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

# Dataset Prepare
def scale_image(data):
    image = tf.cast(data['image'], tf.float32) / 255.0
    return image, data['label']

def prepare_dataset():
    ds = tfds.load('mnist', split='train')
    ds = ds.map(scale_image)
    return ds.cache().shuffle(60000).batch(256).prefetch(tf.data.AUTOTUNE)


# Build Generator
def build_generator(latent_dim, num_classes):
    noise_input = Input((latent_dim,))
    label_input = Input((1,), dtype='int32')
    label_embedding = Embedding(num_classes, latent_dim)(label_input)
    label_embedding = Flatten()(label_embedding)

    x = Multiply()([noise_input, label_embedding])
    x = Dense(7 * 7 * 256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((7, 7, 256))(x)

    # First upsampling
    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Second upsampling
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    output = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    return Model([noise_input, label_input], output)

# Build Discriminator
def build_discriminator(num_classes):
    image_input = Input((28,28,1))
    label_input = Input((1,), dtype='int32')
    label_embedding = Embedding(num_classes, 28*28)(label_input)
    label_embedding = Reshape((28,28,1))(label_embedding)

    x = Concatenate()([image_input, label_embedding])
    for filters in [64, 128, 256]:
        x = Conv2D(filters, 3, strides=2, padding='same')(x); x = LeakyReLU()(x); x = Dropout(0.3)(x)

    x = Flatten()(x); output = Dense(1, activation='sigmoid')(x)
    return Model([image_input, label_input], output)


# Build Classifier
def build_classifier():
    model = Sequential([
        Input((28,28,1)),
        Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
        Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


# cGAN model + train_step
class ConditionalGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_step(self, data):
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]
        latent_dim = self.generator.input[0].shape[-1]

        random_latent = tf.random.normal((batch_size, latent_dim))
        random_labels = tf.random.uniform((batch_size, 1), 0, 10, dtype=tf.int32)
        fake_images = self.generator([random_latent, random_labels], training=False)

        smooth_real = tf.ones((batch_size,1)) * 0.9
        smooth_fake = tf.zeros((batch_size,1)) + 0.1
        y_combined = tf.concat([smooth_real, smooth_fake], axis=0)

        # Train discriminator
        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator([real_images, real_labels], training=True)
            yhat_fake = self.discriminator([fake_images, random_labels], training=True)
            yhat_combined = tf.concat([yhat_real, yhat_fake], axis=0)
            d_loss = self.d_loss_fn(y_combined, yhat_combined)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train generator
        random_latent = tf.random.normal((batch_size, latent_dim))
        random_labels = tf.random.uniform((batch_size, 1), 0, 10, dtype=tf.int32)
        with tf.GradientTape() as g_tape:
            fake_images = self.generator([random_latent, random_labels], training=True)
            fake_preds = self.discriminator([fake_images, random_labels], training=False)
            g_loss = self.g_loss_fn(tf.ones_like(fake_preds), fake_preds)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"g_loss": g_loss, "d_loss": d_loss}


# Monitor and Save Metrics
class ConditionalMonitor(Callback):
    def __init__(self, classifier, latent_dim, num_eval_samples=1000):
        super().__init__()
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.num_eval_samples = num_eval_samples
        self.accuracies = []
        self.f1s = []
        os.makedirs('cgan_images', exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        num_classes = 10
        latent = tf.random.normal((num_classes, self.latent_dim))
        labels = tf.range(num_classes)[:, None]
        images = self.model.generator([latent, labels], training=False)

        fig, axes = plt.subplots(1, num_classes, figsize=(20,2))
        for i, img in enumerate(images):
            axes[i].imshow(img.numpy().squeeze(), cmap='gray')
            axes[i].set_title(str(i)); axes[i].axis('off')
        plt.savefig(f'cgan_images/epoch_{epoch:03}.png'); plt.close()

        latent_eval = tf.random.normal((self.num_eval_samples, self.latent_dim))
        label_eval = tf.random.uniform((self.num_eval_samples, 1), 0, 10, dtype=tf.int32)
        fake_imgs = self.model.generator([latent_eval, label_eval], training=False).numpy()

        y_pred = self.classifier.predict(fake_imgs).argmax(axis=1)
        acc = accuracy_score(label_eval.numpy().squeeze(), y_pred)
        f1 = f1_score(label_eval.numpy().squeeze(), y_pred, average='weighted')
        self.accuracies.append(acc); self.f1s.append(f1)
        print(f"Epoch {epoch}: acc={acc:.4f}, f1={f1:.4f}")

    def on_train_end(self, logs=None):
        df = pd.DataFrame({"epoch": list(range(len(self.accuracies))),
                           "accuracy": self.accuracies, "f1": self.f1s})
        df.to_csv('cgan_metrics.csv', index=False)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy', color='blue')
        ax1.plot(self.accuracies, color='blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('f1', color='red')
        ax2.plot(self.f1s, color='red')
        fig.tight_layout(); fig.savefig('cgan_metrics_plot.png'); plt.close()


# Run training
def main():
    latent_dim = 128
    num_classes = 10
    ds = prepare_dataset()

    generator = build_generator(latent_dim, num_classes)
    discriminator = build_discriminator(num_classes)
    cgan = ConditionalGAN(generator, discriminator)
    cgan.compile(
        g_optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.999),
        d_optimizer=Adam(2e-4, beta_1=0.5, beta_2=0.999),
        g_loss_fn=BinaryCrossentropy(),
        d_loss_fn=BinaryCrossentropy(),
    )

    # Train a classifier
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype('float32')/255.0, -1)
    x_test = np.expand_dims(x_test.astype('float32')/255.0, -1)

    classifier = build_classifier()
    classifier.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=256)

    monitor = ConditionalMonitor(
        classifier=classifier,
        latent_dim=latent_dim,
        num_eval_samples=1000
    )

    cgan.fit(ds, epochs=100, callbacks=[monitor])
    
    # Save models
    os.makedirs('models', exist_ok=True)
    generator.save('models/cgan_generator.keras')
    discriminator.save('models/cgan_discriminator.keras')
    classifier.save('models/mnist_classifier.keras')
    generator = load_model('saved_models/cgan_generator.keras')


if __name__ == "__main__":
    main()