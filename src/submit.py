"""Use kaggle-cli to make submissions."""

import os
from operator import itemgetter

import click
from dotenv import load_dotenv, find_dotenv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_submission(kind='best'):
    """Find the path of a submission file."""
    all_files = os.listdir(os.path.join(THIS_DIR, '..', 'output'))
    all_files = [f for f in all_files if (
        str(f).startswith('subm_') and str(f).endswith('csv'))]

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


@click.command()
@click.option('--kind', type=click.Choice(['best', 'recent']), default='recent')
def make_submission(kind):
    """Make a submission to kaggle using the kaggle-cli tool."""
    # load .env variables for kaggle creds
    load_dotenv(find_dotenv())

    # find submission csv file
    subm_file = find_submission(kind)

    # generate kaggle-cli command
    subm_cmd = 'kg submit {} -u {} -p {} -c {} -m "{}"'.format(
        subm_file,
        os.environ.get('KAGGLE_USER'),
        os.environ.get('KAGGLE_PW'),
        os.environ.get('KAGGLE_COMP'),
        subm_file
    )

    # make system call to kaggle-cli
    os.system(subm_cmd)


if __name__ == '__main__':
    make_submission()
