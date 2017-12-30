"""Use kaggle-cli to make submissions."""

import os
import time
from operator import itemgetter

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_submission(kind='best', ensemble=False, topx=3):
    """Find the path of a submission file."""
    all_files = os.listdir(os.path.join(THIS_DIR, '..', 'output'))
    all_files = [f for f in all_files if (
        str(f).startswith('subm_') and str(f).endswith('csv'))]

    # non-ensembled files only
    all_files = [f for f in all_files if 'ensem' not in f]

    if not ensemble:
        if kind == 'best':
            all_scores = [float(str(f)[28:35]) for f in all_files]
            lowest_loc = min(enumerate(all_scores), key=itemgetter(1))[0]
            lowest_file = all_files[lowest_loc]

            return os.path.abspath(os.path.join(THIS_DIR, '..', 'output', lowest_file))
        else:
            all_times = [str(f)[41:57] for f in all_files]
            latest_loc = max(enumerate(all_times), key=itemgetter(1))[0]
            latest_file = all_files[latest_loc]

            return os.path.abspath(os.path.join(THIS_DIR, '..', 'output', latest_file))
    else:
        if kind == 'best':
            all_scores = [float(str(f)[28:35]) for f in all_files]
            sorted_indicies = np.argsort(all_scores)
            used_indicies = sorted_indicies[:topx]
            used_files = [all_files[i] for i in used_indicies]

            # naive mean "ensembling"
            for f in used_files:
                subm_df = pd.read_csv(os.path.join(THIS_DIR, '..', 'output', f))

                try:
                    ensembled_df = pd.merge(ensembled_df, subm_df, on='id')
                    iceberg_cols = [c for c in ensembled_df.columns if 'ice' in c]
                    ensembled_df['is_iceberg'] = ensembled_df[iceberg_cols].mean(axis=1)
                    ensembled_df = ensembled_df.drop(iceberg_cols, axis=1)
                except:
                    ensembled_df = subm_df.copy()

            # write ensembled results to new csv
            ts = time.strftime("%Y-%m-%d_%H-%M")
            fname = 'subm_ensembled_' + ts + '_top_' + kind + '_' + str(topx) + '.csv'
            ensembled_df.to_csv(os.path.join(THIS_DIR, '..', 'output', fname), index=False)

            return os.path.abspath(os.path.join(THIS_DIR, '..', 'output', fname))
        else:
            all_times = [str(f)[41:57] for f in all_files]
            sorted_indicies = np.argsort(all_times)
            used_indicies = sorted_indicies[-topx:]
            # import pdb; pdb.set_trace()
            used_files = [all_files[i] for i in used_indicies]

            # naive mean "ensembling"
            for f in used_files:
                subm_df = pd.read_csv(os.path.join(THIS_DIR, '..', 'output', f))

                try:
                    ensembled_df = pd.merge(ensembled_df, subm_df, on='id')
                    iceberg_cols = [c for c in ensembled_df.columns if 'ice' in c]
                    ensembled_df['is_iceberg'] = ensembled_df[iceberg_cols].mean(axis=1)
                    ensembled_df = ensembled_df.drop(iceberg_cols, axis=1)
                except:
                    ensembled_df = subm_df.copy()

            # write ensembled results to new csv
            ts = time.strftime("%Y-%m-%d_%H-%M")
            fname = 'subm_ensembled_' + ts + '_top_' + kind + '_' + str(topx) + '.csv'
            ensembled_df.to_csv(os.path.join(THIS_DIR, '..', 'output', fname), index=False)

            return os.path.abspath(os.path.join(THIS_DIR, '..', 'output', fname))


@click.command()
@click.option('--kind', type=click.Choice(['best', 'recent']), default='recent')
@click.option('--ensemble', is_flag=True)
@click.option('--topx', type=int, default=3)
def make_submission(kind, ensemble, topx):
    """Make a submission to kaggle using the kaggle-cli tool."""
    # load .env variables for kaggle creds
    load_dotenv(find_dotenv())

    # find submission csv file
    subm_file = find_submission(kind, ensemble, topx)

    # generate kaggle-cli command
    subm_cmd = 'kg submit {} -u {} -p {} -c {} -m "{}"'.format(
        subm_file,
        os.environ.get('KAGGLE_USER'),
        os.environ.get('KAGGLE_PW'),
        os.environ.get('KAGGLE_COMP'),
        subm_file
    )

    # make system call to kaggle-cli
    print('Making kaggle submission\n' + subm_cmd)
    os.system(subm_cmd)


if __name__ == '__main__':
    make_submission()
