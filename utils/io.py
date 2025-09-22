import fsspec
import torch
import io
import ssl
import urllib.request
import zipfile
import sys
import os.path as osp
from tqdm import tqdm
from datasets_process import ROOT
from typing import Any, Optional, Union
import os
import shutil
from huggingface_hub import hf_hub_download


def torch_safe_save(obj: Any, path: str) -> None:
    if obj is not None:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        with fsspec.open(path, 'wb') as f:
            f.write(buffer.getvalue())


def torch_safe_load(path: str, map_location: Any = None) -> Any:
    if osp.exists(path):
        with fsspec.open(path, 'rb') as f:
            return torch.load(f, map_location)
    return None


def download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None):
    """Downloading URL resources and displaying a progress bar"""

    if filename is None:
        filename = url.rpartition('/')[2].split('?')[0]
    path = osp.join(folder, filename)
    if osp.exists(path):
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path
    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)
    total_size = int(response.headers.get('content-length', 0))

    with fsspec.open(path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=filename, disable=not log
    ) as pbar:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
    return path




def download_hf_file(repo_id, filename, local_dir, subfolder=None, repo_type="dataset", cache_dir=None):
    file_path = os.path.join(local_dir, filename)
    if os.path.exists(file_path):
        print(f"The file {filename} already exists in {local_dir}, skip the download.")
        return file_path
    hf_hub_download(
        repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
        local_dir=local_dir, local_dir_use_symlinks=False, cache_dir=cache_dir, force_download=False
    )
    if subfolder:
        shutil.move(os.path.join(local_dir, subfolder, filename), os.path.join(local_dir, filename))
        shutil.rmtree(os.path.join(local_dir, subfolder))
    return os.path.join(local_dir, filename)


def maybe_log(path: str, log=True):
    if log:
        print('Extracting', path)


def extract_zip(path: str, folder: str, log: bool = True):
    """解压 ZIP 文件，并显示进度条"""
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        file_list = f.namelist()
        with tqdm(total=len(file_list), unit="file", desc="Extracting", disable=not log) as pbar:
            for file in file_list:
                f.extract(file, folder)
                pbar.update(1)


def move_files_in_dir(source_dir: str, target_dir: str):
    """移动目录中的所有文件，并显示进度条"""
    all_files = os.listdir(source_dir)

    with tqdm(total=len(all_files), unit="file", desc="Moving files") as pbar:
        for f in all_files:
            src_path = os.path.join(source_dir, f)
            dst_path = os.path.join(target_dir, f)
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                print(f"Failed to move {src_path} -> {dst_path}. Reason: {e}")
            pbar.update(1)

def delete_folder(folder_dir: str):
    """删除文件夹及其内容，并显示进度条"""
    if osp.isdir(folder_dir):
        all_files = os.listdir(folder_dir)

        with tqdm(total=len(all_files), unit="file", desc="Deleting files") as pbar:
            for filename in all_files:
                file_path = os.path.join(folder_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
                pbar.update(1)
    else:
        print(f"{folder_dir} not exist.")


def delete_processed_files(root: str = ROOT,
                           datasets: Optional[Union[str, list[str]]] = None,
                           delete_processed: bool = True,
                           delete_raw: bool = False):
    """删除数据集中的处理后的文件"""
    if datasets is None:
        datasets = os.listdir(root)
    elif isinstance(datasets, str):
        datasets = [datasets]

    for dataset in datasets:
        path = osp.join(root, dataset)
        if osp.isdir(path):
            task_path = osp.join(path, "task")
            delete_folder(task_path)
            if delete_processed:
                processed_path = osp.join(path, "processed")
                delete_folder(processed_path)
            if delete_raw:
                raw_path = osp.join(path, "raw")
                delete_folder(raw_path)




