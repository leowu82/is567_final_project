"""Helper to download Kaggle datasets using the Kaggle CLI.

Requires `kaggle` to be available in the environment and the user's
`kaggle.json` to be mounted at `/root/.kaggle/kaggle.json` inside the container
or available at `%USERPROFILE%/.kaggle/kaggle.json` on Windows.
"""
import os
import subprocess
import argparse


# Predefined datasets for convenience
DATASETS = {
    'sms': {
        'slug': 'uciml/sms-spam-collection-dataset',
        'name': 'SMS Spam Collection',
        'description': '5,574 SMS messages (87% legitimate, 13% spam)'
    },
    'phishing': {
        'slug': 'shashwatwork/phishing-dataset-for-machine-learning',
        'name': 'Phishing Email Dataset',
        'description': '~82,500 phishing emails'
    },
}


def download(dataset, dest='data/raw'):
    """Download and extract a Kaggle dataset."""
    os.makedirs(dest, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', dest, '--unzip']
    print(f'Downloading: {dataset}')
    print('Running:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        print(f'✓ Downloaded to {dest}/')
    except subprocess.CalledProcessError as e:
        print(f'✗ Download failed: {e}')
        print('Ensure your Kaggle API key is configured at ~/.kaggle/kaggle.json')
        raise


def main():
    p = argparse.ArgumentParser(
        description='Download training datasets for fraud detection system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download SMS Spam Collection
  python data/ingest_kaggle.py --dataset sms
  
  # Download Phishing Email Dataset
  python data/ingest_kaggle.py --dataset phishing
  
  # Download both
  python data/ingest_kaggle.py --dataset sms phishing
  
  # Download by Kaggle slug
  python data/ingest_kaggle.py --dataset uciml/sms-spam-collection-dataset
  
  # List available datasets
  python data/ingest_kaggle.py --list
        '''
    )
    p.add_argument(
        '--dataset', 
        nargs='+',
        help='Dataset name (sms, phishing) or Kaggle slug'
    )
    p.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    p.add_argument(
        '--dest',
        default='data/raw',
        help='Destination directory (default: data/raw)'
    )
    
    args = p.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        print("=" * 60)
        for key, info in DATASETS.items():
            print(f"\n  Name: {key}")
            print(f"  Display: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Slug: {info['slug']}")
        print("\n" + "=" * 60)
        return
    
    if not args.dataset:
        p.print_help()
        return
    
    # Resolve dataset names to slugs
    to_download = []
    for dataset in args.dataset:
        if dataset in DATASETS:
            slug = DATASETS[dataset]['slug']
            name = DATASETS[dataset]['name']
            print(f"Resolving '{dataset}' → {name} ({slug})")
            to_download.append(slug)
        else:
            # Assume it's a direct Kaggle slug
            to_download.append(dataset)
    
    print(f"\nWill download {len(to_download)} dataset(s) to {args.dest}/\n")
    
    for dataset in to_download:
        try:
            download(dataset, args.dest)
        except Exception as e:
            print(f"\nFailed to download {dataset}: {e}")
            continue


if __name__ == '__main__':
    main()
