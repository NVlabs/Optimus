"""
Script to download datasets packaged with the repository.
"""
import os
import argparse

import optimus
import optimus.utils.file_utils as FileUtils
dataset_links = dict(
    Stack="https://drive.google.com/file/d/1ciP5yP0D06gT7QXq7OvxlR-mcepIoHCK/view?usp=drive_link",
    StackThree="https://drive.google.com/file/d/1_Qo0jYPfepe4rUpyDrtbmI9WHPipRwmL/view?usp=drive_link",
    StackFour="https://drive.google.com/file/d/142XBL1Hy2Ru1qGNbv4JlBjHm-Y0q_is2/view?usp=drive_link",
    StackFive="https://drive.google.com/file/d/1z7CFsBEXkj7DxLSpfUteZjPuN2quwBtA/view?usp=drive_link"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Base download directory. Created if it doesn't exist. Defaults to datasets folder in repository.",
    )

    # tasks to download datasets for
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=["Stack"],
        help="Tasks to download datasets for. Defaults to square_d0 task. Pass 'all' to download all tasks\
            for the provided dataset type or directly specify the list of tasks.",
    )

    # dry run - don't actually download datasets, but print which datasets would be downloaded
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="set this flag to do a dry run to only print which datasets would be downloaded"
    )

    args = parser.parse_args()

    # set default base directory for downloads
    default_base_dir = args.download_dir
    if default_base_dir is None:
        default_base_dir = os.path.join(optimus.__path__[0], "../datasets")

    # load args
    download_tasks = args.tasks
    if "all" in download_tasks:
        assert len(download_tasks) == 1, "all should be only tasks argument but got: {}".format(args.tasks)
        download_tasks = list(dataset_links.keys())
    else:
        for task in download_tasks:
            assert task in dataset_links, "got unknown task {}. Choose one of {}".format(task, list(dataset_links.keys()))

    # download requested datasets
    for task in download_tasks:
        download_dir = os.path.abspath(os.path.join(default_base_dir))
        download_path = os.path.join(download_dir, "{}.hdf5".format(task))
        print("\nDownloading dataset:\n    task: {}\n    download path: {}"
            .format(task, download_path))
        url = dataset_links[task]
        if args.dry_run:
            print("\ndry run: skip download")
        else:
            # Make sure path exists and create if it doesn't
            os.makedirs(download_dir, exist_ok=True)
            print("")
            FileUtils.download_url_from_gdrive(
                url=url, 
                download_dir=download_dir,
                check_overwrite=True,
            )
        print("")