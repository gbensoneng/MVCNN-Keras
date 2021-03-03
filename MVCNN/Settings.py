import ast
import os

class Settings:

    def __init__(self, ini_path):
        self.ini_path = ini_path
        lines = []
        with open(self.ini_path) as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            line = line.split('#', 1)[0]
            split = line.strip().split('=')
            arg = split[0]
            value = split[1]
            if arg == "ImageWidth":
                self.image_width = int(value)
            elif arg == "ImageHeight":
                self.image_height = int(value)
            elif arg == "ImageDepth":
                self.image_depth = int(value)
            elif arg == "NumViews":
                self.num_views = int(value)
            elif arg == "MergeMode":
                self.merge_mode = value
            elif arg == 'SVCNNModel':
                self.svcnn_model = value
            elif arg == "BatchSize":
                self.batch_size = int(value)
            elif arg == "NumEpochs":
                self.num_epochs = int(value)
            elif arg == "LearningRate":
                self.lr = float(value)
            elif arg == "Momentum":
                self.momentum = float(value)
            elif arg == "LibraryPath":
                self.library_path = value
        self.class_mapping = {}
        class_id = 0
        for dir in os.listdir(self.library_path):
            self.class_mapping[dir] = class_id
            class_id += 1
        self.num_classes = class_id
        self.input_shape = (self.image_width, self.image_height, self.image_depth)
