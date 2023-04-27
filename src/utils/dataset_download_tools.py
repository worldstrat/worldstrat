import hashlib
import os
import tarfile
from tqdm.auto import tqdm
import pandas as pd
from glob import glob
from pathlib import Path
import requests

def sha256_large_file(file_path, buffer_size=8192):
    """
    Compute the SHA-256 hash of a large file.

    Parameters
    ----------
    file_path : str
        The path to the large file for which the SHA-256 hash should be computed.
    buffer_size : int, optional, default: 8192
        The buffer size (in bytes) used for reading the file in chunks.
        The default value is 8192.

    Returns
    -------
    str
        The hexadecimal representation of the SHA-256 hash.

    Examples
    --------
    >>> file_path = 'path/to/your/large/file.ext'
    >>> hash_result = sha256_large_file(file_path)
    >>> print('SHA-256 hash:', hash_result)
    """

    sha256 = hashlib.sha256()
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        for data in tqdm(iter(lambda: f.read(buffer_size), b''),
                         total=(file_size // buffer_size) + (1 if file_size % buffer_size else 0),
                         unit='blocks', unit_scale=buffer_size, desc=f'Hashing {file_path}'):
            sha256.update(data)
    return sha256.hexdigest()

def md5_large_file(file_path, buffer_size=8192):
    """
    Compute the MD5 hash of a large file.

    Parameters
    ----------
    file_path : str
        The path to the large file for which the MD5 hash should be computed.
    buffer_size : int, optional, default: 8192
        The buffer size (in bytes) used for reading the file in chunks.
        The default value is 8192.

    Returns
    -------
    str
        The hexadecimal representation of the MD5 hash.

    Examples
    --------
    >>> file_path = 'path/to/your/large/file.ext'
    >>> hash_result = md5_large_file(file_path)
    >>> print('MD5 hash:', hash_result)
    """
    md5 = hashlib.md5()
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        for data in tqdm(iter(lambda: f.read(buffer_size), b''),
                         total=(file_size // buffer_size) + (1 if file_size % buffer_size else 0),
                         unit='blocks', unit_scale=buffer_size, desc=f'Hashing {file_path}'):
            md5.update(data)
    return md5.hexdigest()

def check_md5_hash(directory, file_hash_list):
    """
    Check if the MD5 hashes of files in a directory match the expected hashes.

    Parameters
    ----------
    directory : str
        The path to the directory containing the files to be checked.
    file_hash_list : dict
        A dictionary containing file names and their expected MD5 hash.
        Example: {'file1.txt':'expected_md5_hash1', 'file2.txt':'expected_md5_hash2'}

    Returns
    -------
    dict
        A dictionary with file names as keys and a boolean value indicating whether the hash matches.
        Example: {'file1.txt': True, 'file2.txt': False}

    Examples
    --------
    >>> directory = 'path/to/your/directory'
    >>> file_hash_list = {'file1.txt':'expected_md5_hash1', 'file2.txt':'expected_md5_hash2'}
    >>> result = check_md5_hash(directory, file_hash_list)
    >>> print(result)
    """
    result = {}
    for file_name, expected_hash in file_hash_list.items():
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            calculated_hash = md5_large_file(file_path)
            result[file_name] = calculated_hash == expected_hash
        else:
            result[file_name] = False

    return False not in result.values()


def download_large_file(url, destination, buffer_size=8192, max_retries=5):
    """
    Download a large file from a URL with a progress bar, resuming the download in case of a network error.

    Parameters
    ----------
    url : str
        The URL of the large file to download.
    destination : str
        The destination path for the downloaded file.
    buffer_size : int, optional, default: 8192
        The buffer size (in bytes) used for reading the file in chunks.
        The default value is 8192.
    max_retries : int, optional, default: 5
        The maximum number of retries in case of a network error.

    Examples
    --------
    >>> url = 'https://example.com/largefile.ext'
    >>> destination = 'path/to/your/downloaded/largefile.ext'
    >>> download_large_file(url, destination)
    """

    headers = {}
    retries = 0

    if os.path.exists(destination):
        resume_position = os.path.getsize(destination)
        headers['Range'] = f'bytes={resume_position}-'
    else:
        resume_position = 0

    response = requests.get(url, headers=headers, stream=True, timeout=5)

    file_size = int(response.headers.get('content-length', 0)) + resume_position

    with open(destination, 'ab') as f:
        while retries < max_retries:
            try:
                for chunk in tqdm(response.iter_content(chunk_size=buffer_size),
                                 initial=resume_position, total=file_size,
                                 unit='B', unit_scale=True, desc=f'Downloading {destination}',
                                 bar_format='{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt} / {total_fmt}.'):
                    f.write(chunk)
                break
            except requests.exceptions.RequestException:
                retries += 1
                print(f'Retrying download, attempt {retries} of {max_retries}')

    if retries >= max_retries:
        print(f'Download failed after {max_retries} retries.')

def extract_tar_gz(input_path, output_path, num_files=None):
    """
    Extract a .tar.gz archive to the specified output path and display extraction progress.

    This function extracts a .tar.gz archive to a specified output directory while displaying
    the extraction progress, including the number of files extracted, the extraction speed
    (files per second), and the current file being extracted. In case of an interruption or
    error, the extraction can be resumed by re-running the function.

    Parameters
    ----------
    input_path : str
        The path to the .tar.gz archive file to be extracted.
    output_path : str
        The path to the output directory where the files will be extracted.
    num_files : int, optional
        The number of files in the archive. If provided, the progress bar will display the
        total length and progress percentage. Getting the number of files in a .tar.gz archive
        programatically takes a long time (using tar.getmembers()), hence this is used.
        Default is None.

    Returns
    -------
    None

    Examples
    --------
    >>> input_path = 'path/to/your/archive.tar.gz'
    >>> output_path = 'path/to/output/folder'
    >>> extract_tar_gz(input_path, output_path)
    """

    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # Open the tarfile
    with tarfile.open(input_path, 'r:gz') as tar:
        # Create a progress bar with a custom format string
        bar_format = '{desc}: {n_fmt} files | {rate_fmt}{postfix}'
        if num_files is not None:
            bar_format = '{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} files | {rate_fmt}{postfix}'
        
        progress_bar = tqdm(desc=f"Extracting {input_path}", unit="files", total=num_files, bar_format=bar_format)

        # Initialize a list to store exceptions
        exceptions = []

        # Iterate over the files
        for member in tar:
            # Construct the output file path
            output_file_path = os.path.join(output_path, member.name)

            # Check if the file already exists
            if os.path.exists(output_file_path):
                progress_bar.set_postfix_str(f"File already exists: {member.name}", refresh=True)
                progress_bar.update(1)
                continue

            # Extract the current file to the output path
            try:
                tar.extract(member, output_path)
            except Exception as e:
                exceptions.append((member.name, e))
                progress_bar.update(1)
                continue

            # Update the progress bar and display the current file
            progress_bar.set_postfix_str(f"Current file: {member.name}", refresh=True)
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Print the exceptions, if any
        if exceptions:
            print("\nExceptions encountered during extraction:")
            for file_name, exception in exceptions:
                print(f"  - {file_name}: {exception}")
        else:
            print("\nExtraction completed without any exceptions.")



def verify_dataset_structure():
    """
    Verify the structure of the dataset by checking for the presence of required files and folders.

    This function checks if the required files and folders are present in the 'dataset' directory.
    If any parts of the dataset are missing, an exception is raised with information about the
    missing parts. Additionally, it checks if the expected AOIs (areas of interest) are present
    in the 'hr_dataset' and 'lr_dataset' folders.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If any parts of the dataset are missing or the dataset structure is incorrect.

    Examples
    --------
    >>> verify_dataset_structure()
    Dataset structure looks good. ✅
    """

    expected_files = {'hr_dataset', 'lr_dataset', 'metadata.csv', 'stratified_train_val_test_split.csv'}
    dataset_files = set(os.listdir('dataset'))

    if expected_files.intersection(dataset_files) != expected_files:
        print("Dataset parts are missing.")
        raise Exception(f"Please redownload and extract to dataset folder: {expected_files.difference(dataset_files)}")

    metadata = pd.read_csv('dataset/metadata.csv', index_col=0)
    expected_aois = set(metadata.index)

    hr_dataset_aois = set(os.listdir('dataset/hr_dataset/'))
    lr_dataset_l1c_aois = set(Path(aoi).parent.name for aoi in glob('dataset/lr_dataset/*/L1C'))
    lr_dataset_l2a_aois = set(Path(aoi).parent.name for aoi in glob('dataset/lr_dataset/*/L2A'))

    assert expected_aois.intersection(hr_dataset_aois) == expected_aois, f'hr_dataset is missing AOIs, please redownload it.'
    assert expected_aois.intersection(lr_dataset_l1c_aois) == expected_aois, f'lr_dataset (L1C) is missing AOIs, please redownload it.'
    assert expected_aois.intersection(lr_dataset_l2a_aois) == expected_aois, f'lr_dataset (L2A) is missing AOIs, please redownload it.'

    print(f"Dataset structure looks good. ✅")

