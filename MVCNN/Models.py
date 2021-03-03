# region IMPORT
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Maximum, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import vgg16, vgg19, resnet50
from .Settings import Settings
import os


# import tensorflow as tf
# endregion

class MultiViewCNN(Model):

    def __init__(self, settings, svcnns):
        super(MultiViewCNN, self).__init__()
        self.merge_mode = settings.merge_mode
        self.svcnns = svcnns
        self.model_inputs = []
        self.svcnn_outs = []
        for svcnn in svcnns:
            self.model_inputs.append(svcnn.model.input)
            self.svcnn_outs.append(svcnn.model.output)
        # region View Pooling & Output Layers
        if self.merge_mode == 'max':
            self.view_pooling_layer = Maximum(name='view_pool_max')(self.svcnn_outs)
        elif self.merge_mode == 'concat':
            self.view_pooling_layer = Concatenate(axis=1, name='view_pool_concat')(self.svcnn_outs)
        self.x = Dense(512, name='mvcnn_fc1')(self.view_pooling_layer)
        self.model_output = Dense(settings.num_classes, activation='softmax', name='mvcnn_output')(self.x)
        self.model = Model(inputs=self.model_inputs, outputs=self.model_output)
        # endregion

    def call(self, inputs, training=False, **kwargs):
        assert len(inputs) == len(self.inputs)
        for i in range(len(self.inputs)):
            self.inputs[i] = inputs[i]
        return self.model_output

    def generate_model_plot(self, file_name):
        cwd = os.getcwd()
        save_path = os.path.join(cwd, 'results', 'model_diagrams', f'{file_name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_model(self.model, to_file=save_path, show_shapes=True)


class SingleViewCNN(Model):

    def __init__(self, settings: Settings, view_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.view_id = view_id

        # region Use VGG16
        if self.settings.svcnn_model == 'vgg16':
            vgg16_model = vgg16.VGG16(include_top=False, input_shape=settings.input_shape, pooling=None, weights=None)
            x = vgg16_model.layers[-1].output
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(settings.num_classes, activation='softmax', name='predictions')(x)
            self.model_input = vgg16_model.input
            self.model_output = x
            self.model = Model(inputs=self.model_input, outputs=self.model_output)
        # endregion

        # region Use VGG19
        elif self.settings.svcnn_model == 'vgg19':
            vgg19_model = vgg19.VGG19(include_top=False, input_shape=settings.input_shape, pooling=None, weights=None)
            x = vgg19_model.layers[-1].output
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(settings.num_classes, activation='softmax', name='predictions')(x)
            self.model_input = vgg19_model.input
            self.model_output = x
            self.model = Model(inputs=self.model_input, outputs=self.model_output)
        # endregion

        # region Use ResNet50
        elif self.settings.svcnn_model == 'resnet50':
            resnet50_model = resnet50.ResNet50(include_top=False, input_shape=settings.input_shape, pooling=None, weights=None)
            x = resnet50_model.layers[-1].output
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dense(settings.num_classes, activation='softmax', name='probs')(x)
            self.model_input = resnet50_model.input
            self.model_output = x
            self.model = Model(inputs=self.model_input, outputs=self.model_output)
        # endregion

    def remove_top(self):

        remove_layers = 0
        if self.settings.svcnn_model.startswith('vgg'):
            remove_layers = 2
        elif self.settings.svcnn_model == 'resnet50':
            remove_layers = 1
        for i in range(remove_layers):
            self.model._layers.pop()
        for layer in self.model.layers:
            new_name = f"svcnn{self.view_id}_{layer._name}"
            self.rename_layer(layer, new_name)
        self.model_output = self.model.layers[-1].output
        self.model = Model(self.model_input, self.model_output)

    def rename_layer(self, layer, new_name):
        def _get_node_suffix(name):
            for old_name in old_nodes:
                if old_name.startswith(name):
                    return old_name[len(name):]

        old_name = layer.name
        old_nodes = list(self.model._network_nodes)
        new_nodes = []

        for l in self.model.layers:
            if l.name == old_name:
                l._name = new_name
                new_nodes.append(new_name + _get_node_suffix(old_name))
            else:
                new_nodes.append(l.name + _get_node_suffix(l.name))
        self.model._network_nodes = set(new_nodes)

    def generate_model_plot(self, file_name):
        cwd = os.getcwd()
        save_path = os.path.join(cwd, 'results', 'model_diagrams', f'{file_name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_model(self.model, to_file=save_path, show_shapes=True)

    def call(self, inputs, training=False, **kwargs):
        assert len(inputs) == len(self.inputs)
        for i in range(len(self.inputs)):
            self.inputs[i] = inputs[i]
        return self.model_output
