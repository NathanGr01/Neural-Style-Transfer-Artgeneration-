import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import time
from tkinter import filedialog


img_size = 400
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def load_images():
    content_filename = filedialog.askopenfilename(title='Choose Content Image')
    content_image = np.array(Image.open(content_filename).resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    style_filename = filedialog.askopenfilename(title='Choose Style Image')
    style_image =  np.array(Image.open(style_filename).resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    generated_image = tf.Variable(generated_image)
    return (content_image, style_image, generated_image)

def load_pretrained_model():
    vgg = tf.keras.applications.VGG19(include_top=False,
                                    input_shape=(img_size, img_size, 3),
                                    weights='/home/nathangruber/Studium/Master/Coursera/Neural Networks and Deep Learning/Übungsprojekte/NST/pretrained-vgg19-weights.h5')
    vgg.trainable = False
    return vgg

def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]

    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1 / (4 * np.square(n_C) * np.square(n_W * n_H)) * tf.reduce_sum(tf.pow(tf.subtract(GS, GG), 2))
    return J_style_layer

def compute_style_cost(style_image_output, generated_image_output):
    J_style = 0
    _, STYLE_LAYERS = choose_content_and_style_layers()
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]

    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer
    return J_style

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def choose_content_and_style_layers():
    content_layer = [
        ('block1_conv1', 1/3),
        ('block3_conv1', 1/3),
        ('block5_conv1', 1/3),]
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
    return (content_layer, STYLE_LAYERS)

def load_network():
    vgg = load_pretrained_model()
    content_layer, STYLE_LAYERS = choose_content_and_style_layers()
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
    return vgg_model_outputs

def preprocess_images(content_image, style_image, vgg_model_outputs):
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)
    return (a_C, a_S)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

@tf.function()
def train_step(generated_image, vgg_model_outputs, a_C, a_S):
    with tf.GradientTape() as tape:
        a_G = vgg_model_outputs(generated_image)
        
        J_style = compute_style_cost(a_S, a_G)
        J_content = compute_content_cost(a_C, a_G)
        J = total_cost(J_content, J_style, alpha = 10, beta = 40)
        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J

def run_model(epochs = 20001):
    content_image, style_image, generated_image = load_images()
    vgg_model_outputs = load_network()
    a_C, a_S = preprocess_images(content_image, style_image, vgg_model_outputs)
    for i in range(epochs):
        train_step(generated_image, vgg_model_outputs, a_C, a_S)
        if i % 250 == 0:
            print(f"Epoch {i} ")
            image = tensor_to_image(generated_image)
            imshow(image)
            plt.show()
            if i == epochs - 1:
                image.save("output/NST_example.jpg")
            
run_model()
