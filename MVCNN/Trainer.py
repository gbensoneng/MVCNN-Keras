import os
import subprocess
import shutil
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .DataLoader import DataLoader
from .Models import MultiViewCNN, SingleViewCNN
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tqdm.keras import TqdmCallback
from datetime import datetime


class Trainer:

    def __init__(self, settings, model_structure):
        self.settings = settings
        self.model_structure = model_structure
        self.view_mode = None
        if isinstance(self.model_structure, SingleViewCNN):
            self.view_mode = f'svcnn{self.model_structure.view_id}'
        elif isinstance(self.model_structure, MultiViewCNN):
            self.view_mode = 'mvcnn'
        self.model = self.model_structure.model
        self.model_optimizer = SGD(learning_rate=self.settings.lr, momentum=self.settings.momentum)
        self.model_loss = CategoricalCrossentropy()
        self.model_metrics = [categorical_accuracy, mean_squared_error]
        self.train_data_loader = None
        self.test_data_loader = None
        self.train_history = None

    def train(self):
        self.model.compile(optimizer=self.model_optimizer, loss=self.model_loss, metrics=self.model_metrics)
        self.model_structure.generate_model_plot(f'{self.view_mode}_model_diagram')
        self.train_data_loader = DataLoader(self.settings, train_test='train', mode=self.view_mode)
        self.test_data_loader = DataLoader(self.settings, train_test='test', mode=self.view_mode)
        working_dir = os.getcwd()
        now = datetime.now()
        log_path = os.path.join(working_dir, "results", "training_logs", self.view_mode, now.strftime("%m%d%Y%H%M%S"))
        print(f'\nBeginning training, to monitor open TensorBoard by running the following command: tensorboard --logdir="{log_path}"\n')
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1, update_freq='batch')
        checkpoint_filepath = os.path.join(working_dir, "results", "models", f"{self.view_mode}_best_model.h5")
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                              monitor='val_categorical_accuracy', mode='max', save_best_only=True)
        early_stopping_callback = EarlyStopping(monitor='categorical_accuracy', min_delta=0.005, patience=20,
                                                mode='max')
        tqdm_callback = TqdmCallback(verbose=0)
        self.model.fit(x=self.train_data_loader,
                       epochs=self.settings.num_epochs,
                       verbose=0,
                       validation_data=self.test_data_loader,
                       callbacks=[tensorboard_callback,
                                  checkpoint_callback,
                                  early_stopping_callback,
                                  tqdm_callback])
