# region Import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .Models import MultiViewCNN, SingleViewCNN
from .Trainer import Trainer
from .Settings import Settings
# endregion


def train_model():

    cwd = os.getcwd()
    settings_path = os.path.join(cwd, "settings.ini")
    settings = Settings(settings_path)
    svcnns = []
    for i in range(settings.num_views):
        print(f'Training SVCNN for view {i+1} out of {settings.num_views}\n')
        svcnn = SingleViewCNN(settings, view_id=i)
        trainer = Trainer(settings, svcnn)
        trainer.train()
        svcnn.remove_top()
        svcnns.append(svcnn)
    mvcnn = MultiViewCNN(settings, svcnns)
    mvcnn.generate_model_plot('mvcnn_model_diagram')
    trainer = Trainer(settings, mvcnn)
    trainer.train()
    print("---------- TRAINING COMPLETE ----------")


if __name__ == '__main__':
    train_model()
