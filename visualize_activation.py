import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Normalizing image would make a huge difference
def normalize(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)


def sort_features(feature_map):
    idx = np.argsort(-1 * np.sum(feature_map, axis=(0, 1)))
    feature_map = feature_map[:, :, idx]
    return feature_map

def create_submodel(model, layers_name):
    if type(layers_name) is str:
        layers_name = [layers_name]
    submodel = Model([model.inputs[0]], [model.get_layer(layer_name).output for layer_name in layers_name])
    return submodel

def view_layer(feature_maps, layers_name, input_img, loc):
    if type(feature_maps) is not list:
        feature_maps = [feature_maps]
        layers_name = [layers_name]
    for i in range(len(layers_name)):
        feature_map = feature_maps[i][0].numpy()
        layer_name = layers_name[i]
        feature_map = sort_features(feature_map)
        # create an empty canvas
        feature_size = 4 * np.array(feature_map[:, :, 0].shape)
        features = np.zeros(feature_size)
        activation_size = 4 * np.array(input_img.shape[1:3])
        activation = np.zeros(activation_size)
        img_gray = rgb2gray(input_img[0]) / 255.
        h, w = feature_map.shape[0], feature_map.shape[1]
        H, W = img_gray.shape[0], img_gray.shape[1]
        for i in range(16):
            x = i % 4
            y = i // 4
            features[y * h:(y + 1) * h, x * w:(x + 1) * w] = feature_map[:, :, i]
            feature_map_resized = resize(normalize(feature_map[:, :, i]), input_img.shape[1:3])
            activation[y * H:(y + 1) * H, x * W:(x + 1) * W] = feature_map_resized * img_gray
        plt.figure()
        plt.title(f"feature maps: {layer_name}")
        plt.axis("off")
        plt.imshow(features)
        plt.savefig(f"output/{loc}_feature_{layer_name}.png")
        # plt.show()

        plt.figure()
        plt.title(f"activation {layer_name}")
        plt.imshow(activation)
        plt.axis("off")
        plt.savefig(f"output/{loc}_activation_{layer_name}.png")
        plt.show()

def view_detection(feature_maps,loc):
    # show the object being detected in by the last convNet layer in summation
    feature_sum = np.sum(feature_maps[-1][0], axis=-1)
    feature_sum_resize = resize(normalize(feature_sum), (65, 320))
    output = feature_sum_resize[..., np.newaxis] * img[0][70:-25,:,:]
    plt.figure()
    plt.imshow(feature_sum)
    plt.savefig(f"output/{loc}_feature_sum.png")
    plt.figure()
    plt.imshow(output.astype(np.uint8))
    plt.savefig(f"output/{loc}_detection_sum.png")
    plt.show()
    return output


if __name__ == '__main__':
    # load model
    model = load_model("model-best1.h5")
    layers_name = [layer.name for layer in model.layers if "conv" in layer.name]
    submodel = create_submodel(model, layers_name)
    # IMAGE_PATH = 'data/right_2020_12_12_19_40_12_084.jpg'  # or 'data/cat.jpg'
    final = np.zeros((65, 320,3))
    for path in os.listdir("./data"):
        img = load_img("./data/"+path)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        loc = path.split("_")[0]
        feature_maps = submodel(img)
        # show the feature map and activation detected by each convNet layer
        view_layer(feature_maps, layers_name, img,loc)
        output = view_detection(feature_maps,loc)
        final += output


    final = normalize(final)
    plt.imshow(final)
    plt.savefig(f"output/final_detection.png")
    plt.show()