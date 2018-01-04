"""Use trained models to make predictions."""

import os
import time
from operator import itemgetter

import click
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_model(kind='best'):
    all_files = os.listdir(os.path.join(THIS_DIR, '..', 'weights'))
    all_files = [f for f in all_files if (
        str(f).startswith('model_') and str(f).endswith('pth'))]

    if kind == 'best':
        all_scores = [float(str(f)[23:-4]) for f in all_files]
        lowest_loc = min(enumerate(all_scores), key=itemgetter(1))[0]
        lowest_file = all_files[lowest_loc]

        return os.path.abspath(os.path.join(THIS_DIR, '..', 'weights', lowest_file))
    else:
        all_times = [str(f)[6:22] for f in all_files]
        latest_loc = max(enumerate(all_times), key=itemgetter(1))[0]
        latest_file = all_files[latest_loc]

        return os.path.abspath(os.path.join(THIS_DIR, '..', 'weights', latest_file))


@click.command()
@click.option('--kind', type=click.Choice(['best', 'recent']), default='recent')
def make_predictions(kind):
    # get model
    model_file = find_model(kind)
    net = torch.load(model_file)

    print('Predicting with ' + model_file)

    if USE_CUDA:
        net.cuda()

    net.eval() # set eval for inference

    # read in testing data and prep df of predictions
    test_file = os.path.join(THIS_DIR, '..', 'input', 'test.json')
    full_test = pd.read_json(test_file)

    full_test['inc_angle'] = pd.to_numeric(full_test['inc_angle'],
                                           errors='coerce')
    full_test['band_1'] = full_test['band_1'].apply(
        lambda x: np.array(x).reshape(75, 75))
    full_test['band_2'] = full_test['band_2'].apply(
        lambda x: np.array(x).reshape(75, 75))

    columns = ['id', 'is_iceberg']
    df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)

    # make prediction for each image
    for idx, row in full_test.iterrows():
        # get 2 channels of our image
        img1 = row['band_1']
        img2 = row['band_2']
        img = np.stack([img1, img2], axis=2)

        # reshape image for pytorch
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)

        # add additional dimension since not passing data in batches as in training
        img = img[np.newaxis, :]

        # convert image to tensor
        img = torch.from_numpy(img)

        if USE_CUDA:
            # set volatile=True to avoid unecessary memory usage for computing gradients
            img = Variable(img.cuda(), volatile=True)
        else:
            img = Variable(img, volatile=True)

        # make prediction
        pred = ((net(img).data).float()).cpu().numpy().item()

        # append dataframe
        df_pred = df_pred.append(
            {
                'id': row['id'],
                'is_iceberg': pred
            },
            ignore_index=True
        )

    # because we are using LogLoss, we are penalized for "over confidence"
    # so, clip the predictions so not as close to 0 or 1
    df_pred['is_iceberg'] = np.clip(df_pred['is_iceberg'], 0.015, 0.985)

    # save prediction file
    ts = time.strftime("%Y-%m-%d_%H-%M")
    model_file = str(str(model_file).split('weights')[1]).replace('.pth', '').replace('\\', '').replace('/', '')
    fname = 'subm_' + model_file + '_pred_' + ts + '.csv'

    df_pred.to_csv(os.path.join(THIS_DIR, '..', 'output', fname),
                   index=False, float_format='%.6f')


if __name__ == '__main__':
    make_predictions()
