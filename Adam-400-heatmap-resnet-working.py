import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from heatmap_gen import *
import numpy as np
import warnings
import os
import os.path
import glob
from keras.layers import Input
from vis.visualization import visualize_cam
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
np.random.seed(1)


def plot_acc(history, path):
    figure = plt.gcf()
    figure.set_size_inches(24, 9)
    ax = plt.subplot()
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    colors = iter(colormap.gist_rainbow(np.linspace(0, 1, len(history))))
    for i in range(len(history)):
        color=next(colors)
        plt.plot(history[i].history['acc'], label='Train '+str(i), color=color, linestyle='solid')
        plt.plot(history[i].history['val_acc'], label='Test '+str(i), color=color, linestyle='dotted')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig(path)


def plot_loss(history, path):
    figure = plt.gcf()
    figure.set_size_inches(24, 9)
    ax = plt.subplot()
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    colors = iter(colormap.gist_rainbow(np.linspace(0, 1, len(history))))
    for i in range(len(history)):
        color=next(colors)
        plt.plot(history[i].history['loss'], label='Train '+str(i), color=color, linestyle='solid')
        plt.plot(history[i].history['val_loss'], label='Test '+str(i), color=color, linestyle='dotted')
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig(path)


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def adam_400epochs(path, nome_teste, p):

    imagedir = path
    cur_dir = os.getcwd()
    os.chdir(imagedir)  # the parent folder with sub-folders

    # Get number of samples per family
    list_fams = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names

    if list_fams[0] == '.DS_Store':
        list_fams.pop(0)

    no_imgs = []  # No. of samples per family
    for i in range(len(list_fams)):
        os.chdir(list_fams[i])
        len1 = len(glob.glob('*.png'))  # assuming the images are stored as 'png'
        no_imgs.append(len1)
        os.chdir('..')
    num_samples = np.sum(no_imgs)  # total number of all samples

    # Compute the labels
    y = np.zeros(num_samples)
    pos = 0
    label = 0
    for i in no_imgs:
        print("Label:%2d\tFamily: %15s\tNumber of images: %d" % (label, list_fams[label], i))
        for j in range(i):
            y[pos] = label
            pos += 1
        label += 1
    num_classes = label

    # Compute the features
    width, height, channels = (224, 224, 3)
    X = np.zeros((num_samples, width, height, channels))
    cnt = 0
    list_paths = []  # List of image paths
    print("Processing images ...")
    for i in range(len(list_fams)):
        for img_file in glob.glob(list_fams[i] + '/*.png'):
            print("[%d] Processing image: %s" % (cnt, img_file))
            list_paths.append(os.path.join(os.getcwd(), img_file))
            img = image.load_img(img_file, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            X[cnt] = x
            cnt += 1
    print("Images processed: %d" % cnt)

    os.chdir(cur_dir)

    # Encoding classes (y) into integers (y_encoded) and then generating one-hot-encoding (Y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    Y = np_utils.to_categorical(y_encoded)

    ############################################## Creating ResNet50 Architecture #############################

    image_shape = (224, 224, 3)
    weights='imagenet'
    input_shape=image_shape
    include_top=False
    input_tensor=None
    pooling=None
    classes=1000


    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model_layers = x
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
'at ~/.keras/keras.json.')

#####################################################################################################

    dataset_main_folder = imagedir.rsplit('/', 1)
    feature_folder = dataset_main_folder[0] + '/features/'
    filename = feature_folder + nome_teste + '-adam-400-epochs.npy'
    if os.path.exists(filename):
        print("Loading ResNet50 extracted features from %s ..." % filename)
        resnet50features = np.load(filename)
    else:
        print("Extracting features from ResNet50 layers ...")
        resnet50features = model.predict(X)
        print("Saving ResNet50 extracted features into %s ..." % filename)
        os.makedirs(feature_folder)
        np.save(filename, resnet50features)

    # Create stratified k-fold subsets
    kfold = 10  # no. of folds
    skf = StratifiedKFold(kfold, shuffle=True, random_state=1)
    skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
    cnt = 0
    for index in skf.split(X, y):
        skfind[cnt] = index
        cnt += 1

    # Training top_model and saving min training loss weights
    num_epochs = 4
    history = []
    conf_mat = np.zeros((len(list_fams), len(list_fams)))  # Initializing the Confusion Matrix
    top_model = None
    filenametopweights = feature_folder + 'weights-' + nome_teste + '-adam-400-epochs.h5'
    checkpointer = ModelCheckpoint(filepath=filenametopweights, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
    for i in range(kfold):
        train_indices = skfind[i][0]
        test_indices = skfind[i][1]
        X_train = resnet50features[train_indices]
        Y_train = Y[train_indices]
        X_test = resnet50features[test_indices]
        Y_test = Y[test_indices]
        y_test = y[test_indices]

        top_input = Input(shape=resnet50features.shape[1:])
        x = GlobalAveragePooling2D(name='avg_pool')(top_input)
        predict = Dense(num_classes, activation='softmax', name='predictions')(x)
        top_model = Model(input=top_input, output=predict)
        top_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        h = top_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs,
                          batch_size=X_train.shape[0], verbose=1, callbacks=[checkpointer])

        history.append(h)

        y_prob = top_model.predict(X_test, verbose=0)  # Testing
        y_pred = np.argmax(y_prob, axis=1)
        print("[%d] Test acurracy: %.4f" % (i, accuracy_score(y_test, y_pred)))
        cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix for this fold
        conf_mat = conf_mat + cm  # Compute global confusion matrix

    # Computing the average accuracy
    avg_acc = np.trace(conf_mat) / sum(no_imgs)
    print("\nAverage acurracy: %.4f" % avg_acc)

    # Ploting the confusion matrix
    conf_mat = conf_mat.T  # since rows and cols are interchangeable
    conf_mat_norm = conf_mat / no_imgs  # Normalizing the confusion matrix

    conf_mat = np.around(conf_mat_norm, decimals=2)  # rounding to display in figure
    figure = plt.gcf()
    figure.set_size_inches(14, 10)
    plt.imshow(conf_mat, interpolation='nearest')
    for row in range(len(list_fams)):
        for col in range(len(list_fams)):
            plt.annotate(str(conf_mat[row][col]), xy=(col, row), ha='center', va='center')
    plt.xticks(range(len(list_fams)), list_fams, rotation=90, fontsize=10)
    plt.yticks(range(len(list_fams)), list_fams, fontsize=10)
    plt.title(path + 'Adam  400 epochs')
    plt.colorbar()
    path_resultados = "../resultados/" + nome_teste
    if not os.path.exists(path_resultados):
        os.makedirs(path_resultados)
    plt.savefig(path_resultados + "/Adam-400epochs.jpg")
    plt.cla()
    plt.clf()

    plot_acc(history, path_resultados + "/Adam-400epochs-acc.jpg")
    plt.cla()
    plt.clf()
    plot_loss(history, path_resultados + "/Adam-400epochs-loss.jpg")
    plt.cla()
    plt.clf()


    ############### Stacking top model ################
    # top_input = Input(shape=resnet50features.shape[1:])
    # x = GlobalAveragePooling2D(name='final_pool')(model_layers)
    x = Dense(num_classes, activation='softmax', name='predictions')(model_layers)
    model = Model(inputs, x, name='resnet50')
    model.load_weights(filenametopweights, by_name=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
    numimgheatmap = 1
    path_txt = path_resultados + "/" + nome_teste + "-PredClass.txt"
    txt_file = open(path_txt, 'w')
    for fam in list_fams:
        path_resultados_orig = path_resultados
        print("Class: %s" %(fam))
        fam_samples = [name for name in list_paths if fam in name.split('/')[-2:-1]]
        print(len(fam_samples))
        image_paths = fam_samples

        heatmaps = []
        myfig = plt.figure()
        plt.axis('off')

        path_resultados = path_resultados_orig + "/heatmaps/" + fam + "/"
        
        if not os.path.exists(path_resultados):
            os.makedirs(path_resultados)
        for path in image_paths:
            seed_img = image.load_img(path, target_size=(224, 224))
            seed_img = image.img_to_array(seed_img)
            img_orig = seed_img
            seed_img = np.expand_dims(seed_img, axis=0)
            pred_class = np.argmax(model.predict(seed_img))
            aux = path.split('/')
            imagem = aux[len(aux) - 1]
            heatMapName = path_resultados + "heatmap-" + imagem[:-4] + ".png"
            heatMapNameArray = path_resultados + "HMarr-" + imagem[:-4] + ".npy"
            print(len(model.layers))
            heatmap = visualize_class_activation_map(model, path, heatMapName)
            # heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)

            np.save(heatMapNameArray, heatmap)
            plt.imsave(fname=heatMapName, arr=heatmap)

            orig = np.array(img_orig, dtype=np.float)
            orig /= 255
            heatmap = np.array(heatmap, dtype=np.float)
            heatmap /= 255
            a_channel = np.ones(orig.shape, dtype=np.float) / 3.7
            image_final = (heatmap * a_channel) + (orig * (1 - a_channel))
            txt_file.write("Image: %s - Class: %s - Pred: %d (%s)\n" % (path.split('/')[-1:], path.split('/')[-2:-1],
                                                                      pred_class, list_fams[pred_class]))

            print("Image: %s" % (path.split('/')[-1:]))
            heatMapNameMerge = path_resultados + "/heatmap-" + imagem[:-4] + "-merge.png"
            plt.imsave(fname=heatMapNameMerge, arr=image_final)
        # img_data = cv2.cvtColor(utils.stitch_images(heatmaps), cv2.COLOR_BGR2RGB)

        # plt.imshow(img_data)
        # plt.show()


path = "./teste/database-09-DSI"
nome_teste = "teste"
imgteste = "testess"

 
adam_400epochs(path, nome_teste, imgteste)