from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from tqdm import tqdm
import random


def generate_multi_view_objects(library_dir, class_mapping, num_views):

    multi_view_objects = []
    image_paths = [os.path.join(root, f)
                   for root, dirs, files
                   in os.walk(library_dir)
                   for f in files
                   if f.endswith(".png")]

    prev_class_name = None
    prev_obj_id = None
    prev_obj = None

    for image_path in tqdm(image_paths, 'Generating multi-view objects'):
        image_view = ImageView(image_path)
        if image_view.view_num >= num_views:
            continue
        object_id = image_view.obj_id
        class_name = image_view.class_name
        is_same_obj = (object_id == prev_obj_id) and (class_name == prev_class_name)
        if not is_same_obj:
            class_id = class_mapping[class_name]
            multi_view_object = MultiViewObject(class_id, class_name, object_id)
            multi_view_objects.append(multi_view_object)
        else:
            multi_view_object = prev_obj
        multi_view_object.image_views.append(image_view)
        prev_class_name = class_name
        prev_obj_id = object_id
        prev_obj = multi_view_object
    return multi_view_objects


class DataLoader(Sequence):

    def __init__(self, settings, train_test, mode='mvcnn'):
        self.mode = mode
        self.library_path = settings.library_path
        self.num_classes = settings.num_classes
        self.batch_size = settings.batch_size
        self.num_views = settings.num_views
        self.input_shape = settings.input_shape
        self.train_test = train_test
        self.multi_view_objects = generate_multi_view_objects(self.library_path, settings.class_mapping, settings.num_views)
        self.num_objs = len(self.multi_view_objects)

    @classmethod
    def from_dir(cls, dir_path, settings):
        settings.library_path = dir_path
        settings.batch_size = 1
        return cls(settings, 'test', 'mvcnn')

    def __len__(self):
        dataset_size = len(self.multi_view_objects)
        return int(np.floor(dataset_size / self.batch_size))

    def __getitem__(self, index):

        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        indexes = range(start, stop)
        x, y = self.__data_generation(indexes)
        return x, y, [None]

    def __data_generation(self, indexes):

        x_batch = None
        view_id = None
        if self.mode == 'mvcnn':
            x_batch = [np.empty((self.batch_size, *self.input_shape))] * self.num_views
        elif self.mode.startswith('svcnn'):
            x_batch = [np.empty((self.batch_size, *self.input_shape))]
            view_id = int(self.mode.split('svcnn')[-1])
        y_batch = np.empty((self.batch_size, self.num_classes), dtype=int)
        batch_index = 0
        for i in indexes:
            multi_view_object = self.multi_view_objects[i]
            x_batch, y_batch = multi_view_object.to_tensor(self.input_shape, self.num_classes, view_id, x_batch, y_batch, batch_index)
            batch_index += 1
        return x_batch, y_batch

    def on_epoch_end(self):

        random.shuffle(self.multi_view_objects)


class ImageView:

    def __init__(self, image_path):
        self.image_path = image_path
        file_name = os.path.basename(self.image_path).split(".png")[0]
        file_split = file_name.split('_')
        self.class_name = file_split[0]
        self.obj_id = int(file_split[1].split(".obj")[0])
        self.view_num = int(file_split[3].split("v")[1])


class MultiViewObject:

    def __init__(self, class_id, class_name, obj_id):
        self.class_id = class_id
        self.class_name = class_name
        self.obj_id = obj_id
        self.image_views = []

    def to_tensor(self, input_shape, num_classes, view_id=None, x_batch=None, y_batch=None, batch_index=None):
        x = []
        specific_view = view_id is not None
        for i in range(len(self.image_views)):
            if specific_view and view_id != i:
                continue
            image_view = self.image_views[i]
            img = load_img(image_view.image_path, target_size=input_shape)
            img_arr = img_to_array(img)
            x.append(img_arr)
        y = to_categorical(self.class_id, num_classes)
        if x_batch is None and y_batch is None:
            return x, y
        else:
            view_loop = []
            if specific_view:
                view_loop.append(0)
            else:
                view_loop = range(len(self.image_views))
            for j in view_loop:
                x_batch[j][batch_index, ] = x[j]
                y_batch[batch_index] = y
                # batch_index += 1
        return x_batch, y_batch
