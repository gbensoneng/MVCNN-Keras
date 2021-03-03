import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .Settings import Settings
from .Models import MultiViewCNN, SingleViewCNN
from .DataLoader import generate_multi_view_objects
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_model(settings, model_path=None):
    svcnns = []
    for i in range(settings.num_views):
        print(f'Loading SVCNN weights for view {i+1} of {settings.num_views}\n')
        svcnn = SingleViewCNN(settings, view_id=i)
        svcnn.remove_top()
        svcnns.append(svcnn)
    mvcnn = MultiViewCNN(settings, svcnns)
    cwd = os.getcwd()
    if model_path is None:
        model_path = os.path.join(cwd, 'results', 'models', 'mvcnn_best_model.h5')
    mvcnn.model.load_weights(model_path)
    return mvcnn


def predict_objects(mvcnn, multi_view_objects, settings):
    for mvo in tqdm(multi_view_objects, 'Creating predictions for multi view objects...'):
        true_class = mvo.class_name
        mvx, mvy = mvo.to_tensor(settings.input_shape, settings.num_classes)
        data_in = [np.expand_dims(x, axis=0) for x in mvx]
        prediction = mvcnn.model.predict(data_in)[0]
        confidence = np.max(prediction)
        pred_class_id = np.argmax(prediction)
        pred_class = None
        for key, value in settings.class_mapping.items():
            if value == pred_class_id:
                pred_class = key
                break
        yield true_class, pred_class, confidence


def test_model(images_dir, model_path=None):
    cwd = os.getcwd()
    settings_path = os.path.join(cwd, "settings.ini")
    settings = Settings(settings_path)
    mvcnn = load_model(settings, model_path)
    multi_view_objects = generate_multi_view_objects(images_dir, settings.class_mapping, settings.num_views)
    results_df = pd.DataFrame(columns=['True Class', 'Predicted Class', 'Confidence'])
    for true_class, pred_class, confidence in predict_objects(mvcnn, multi_view_objects, settings):
        results_df.loc[len(results_df)] = [true_class, pred_class, confidence]
    results_df.to_csv(os.path.join(cwd, 'results', "results.csv"))

    print("---------- PREDCITIONS COMPLETE ----------")


if __name__ == '__main__':
    images_dir = r""
    test_model(images_dir)
