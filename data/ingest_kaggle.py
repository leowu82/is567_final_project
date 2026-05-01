"""Helper to download Kaggle datasets using the Kaggle CLI.

Requires `kaggle` to be available in the environment and the user's
`kaggle.json` to be mounted at `/root/.kaggle/kaggle.json` inside the container
or available at `%USERPROFILE%/.kaggle/kaggle.json` on Windows.
"""
import os
import subprocess
import argparse

def download(dataset, dest='data/raw'):
    os.makedirs(dest, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', dest, '--unzip']
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Kaggle dataset slug, e.g. uciml/sms-spam-collection-dataset')
    args = p.parse_args()
    download(args.dataset)

if __name__ == '__main__':
    main()
