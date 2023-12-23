from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from imageio import imread
import platform


def load_pickle(f):
    """ 用于加载pickle文件，并根据Python版本进行适当的处理 """
    # 获取当前Python解释器的主版本、次版本和微版本号，并将其存储在 version 变量中。
    version = platform.python_version_tuple()
    # 检查主版本号是否为2。在Python 2.x 版本中，pickle 模块不需要指定编码。
    if version[0] == "2":
        # 如果是Python 2.x 版本，直接使用 pickle.load 加载pickle文件。
        return pickle.load(f)
    # 如果是Python 3.x 版本，使用 pickle.load 加载pickle文件，并通过 encoding="latin1" 指定编码
    elif version[0] == "3":
        # 因为在Python 3.x 版本中，pickle 默认使用UTF-8编码，而在CIFAR-10数据集中的批次文件使用的是Latin-1编码。
        return pickle.load(f, encoding="latin1")
    # 如果主版本号既不是2也不是3，则抛出一个值错误，表示Python版本无效。
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ 加载CIFAR-10数据集中的单个批次 """
    # 使用 with 语句打开指定的文件，以二进制只读模式(rb)读取文件内容。这里 filename 是批次文件的路径(data_batch_%d)。
    with open(filename, "rb") as f:
        # 加载文件内容，将其存储在 datadict 变量中。
        datadict = load_pickle(f)
        # 从 datadict 中获取键为 "data" 的值，即图像数据。
        X = datadict["data"] # X.shape: (10000, 3072)
        # print("X.shape: ",X.shape)
        # 从 datadict 中获取键为 "labels" 的值，即标签数据。
        Y = datadict["labels"]
        # 对图像数据进行形状变换和轴变换：
        # 首先，使用 reshape 将图像数据的形状调整为 (10000, 3, 32, 32)，表示有10000张图像，每张图像有3个通道，大小为32x32。
        # 然后，使用 transpose 进行轴变换，将通道维度移到最后的位置，变为 (10000, 32, 32, 3)。
        # 最后，使用 astype("float") 将数据类型转换为浮点型。
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float") 
        # 将标签数据转换为NumPy数组。
        # print("type of Y:",type(Y))
        Y = np.array(Y) # type of Y: <class 'list'>
        return X, Y


def load_CIFAR10(ROOT): # ROOT: 'cs231n/datasets/cifar-10-batches-py'
    """ 加载CIFAR-10数据集 """
    # 初始化两个空列表，分别用于存储训练数据和标签
    xs = []
    ys = []

    for b in range(1, 6):
        # 构建当前批次文件的路径。os.path.join 用于连接目录和文件名
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        # 加载当前批次的数据和标签
        X, Y = load_CIFAR_batch(f)
        # 将加载的数据和标签分别添加到 xs 和 ys 列表中。
        xs.append(X)
        ys.append(Y)
    
    # 使用 numpy.concatenate 将 xs 中的所有数据堆叠在一起，形成训练数据集 Xtrain
    Xtrain = np.concatenate(xs)
    # 使用 numpy.concatenate 将 ys 中的所有标签堆叠在一起，形成训练标签集 Ytrain
    Ytrain = np.concatenate(ys)
    # 删除不再需要的 X 和 Y 变量，以释放内存
    del X, Y
    # 加载测试数据集与测试标签集
    Xtest, Ytest = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    # 返回训练数据集、训练标签集、测试数据集和测试标签集
    return Xtrain, Ytrain, Xtest, Ytest


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(
        os.path.dirname(__file__), "datasets/cifar-10-batches-py"
    )
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, "wnids.txt"), "r") as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, "words.txt"), "r") as f:
        wnid_to_words = dict(line.split("\t") for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(",")]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print("loading training data for synset %d / %d" % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, "train", wnid, "%s_boxes.txt" % wnid)
        with open(boxes_file, "r") as f:
            filenames = [x.split("\t")[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, "train", wnid, "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, "val", "val_annotations.txt"), "r") as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split("\t")[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, "val", "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, "test", "images"))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, "test", "images", img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, "test", "test_annotations.txt")
    if os.path.isfile(y_test_file):
        with open(y_test_file, "r") as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split("\t")
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
        "class_names": class_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "mean_image": mean_image,
    }


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            try:
                models[model_file] = load_pickle(f)["model"]
            except pickle.UnpicklingError:
                continue
    return models


def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet.

    Inputs:
    - num: Number of images to load (max of 25)

    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = os.path.join(
        os.path.dirname(__file__), "datasets/imagenet_val_25.npz"
    )
    if not os.path.isfile(imagenet_fn):
        print("file %s not found" % imagenet_fn)
        print("Run the following:")
        print("cd cs231n/datasets")
        print("bash get_imagenet_val.sh")
        assert False, "Need to download imagenet_val_25.npz"

    # modify the default parameters of np.load
    # https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    f = np.load(imagenet_fn)
    np.load = np_load_old
    X = f["X"]
    y = f["y"]
    class_names = f["label_map"].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names
