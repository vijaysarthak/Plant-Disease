"""
download_data.py — downloads the three image datasets.

    PlantVillage -> data/datasets/plantvillage   (lab photos, with git, only the colour images)
    PlantDoc     -> data/datasets/plantdoc       (field photos, as a ZIP file, see below)
    Rice         -> data/datasets/rice           (a crop the model never sees, with git)

Why PlantDoc is a ZIP and not `git clone`:
    87 PlantDoc photos have names like  "IMG_1629.JPG?1507122477.jpg"  (with a "?"), and a few names
    are so long that the full path is over 260 characters. Windows allows neither, so `git clone`
    stops with "exit status 128". From a ZIP we unpack the files ourselves and give those photos a
    safe name ("?" -> "_", long names shortened). The photos themselves are not changed.

If a dataset is already complete, it is NOT downloaded again.
If a folder is only half there (an earlier download failed), it is deleted and downloaded again.
Used by: the notebook (Step 3).
"""
import os
import shutil
import stat
import subprocess
import urllib.request
import zipfile

from config import DATASETS_DIR, PLANTVILLAGE_DIR, PLANTDOC_DIR, RICE_DIR
from config import PLANTVILLAGE_URL, PLANTDOC_ZIP_URL, RICE_URL

CHARACTERS_NOT_ALLOWED_ON_WINDOWS = '<>:"\\|?*'


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def run(command, folder):
    """Run a terminal command inside `folder`. If it fails, show git's own error message."""
    print("$", command)
    result = subprocess.run(command, shell=True, cwd=folder, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"This command failed (exit code {result.returncode}): {command}")


def delete_folder(folder):
    """Delete a folder completely. (On Windows, git makes some files read-only; we unlock them first.)"""
    def unlock_and_delete(function, path, error):
        os.chmod(path, stat.S_IWRITE)
        function(path)
    if folder.exists():
        print("Deleting the incomplete folder", folder)
        shutil.rmtree(folder, onerror=unlock_and_delete)


def safe_file_name(name, max_length=80):
    """Make a file name that Windows accepts: replace forbidden characters and shorten long names."""
    for character in CHARACTERS_NOT_ALLOWED_ON_WINDOWS:
        name = name.replace(character, "_")
    stem, extension = os.path.splitext(name)
    if len(name) > max_length:
        stem = stem[:max_length - len(extension)]
    return stem + extension


def download_file(url, destination):
    """Download one big file and print the progress every 100 MB."""
    print("Downloading", url)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as file:
        downloaded = 0
        next_report = 100
        while True:
            chunk = response.read(1024 * 1024)          # 1 MB at a time
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if downloaded / 1e6 >= next_report:
                print(f"  {downloaded / 1e6:.0f} MB")
                next_report += 100
    print(f"  done: {downloaded / 1e6:.0f} MB")


def unzip_plantdoc(zip_path, target_folder):
    """Unpack only the train/ and test/ photos, giving every photo a Windows-safe name."""
    renamed = 0
    used_paths = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            # "PlantDoc-Dataset-master/train/Apple leaf/photo.jpg" -> ["train", "Apple leaf", "photo.jpg"]
            parts = member.filename.split("/")[1:]
            if len(parts) != 3 or parts[0] not in ("train", "test"):
                continue                                  # skip README, LICENSE, ...
            part, class_folder, file_name = parts

            new_name = safe_file_name(file_name)
            # two long names can become the same short name -> add a number
            number = 1
            stem, extension = os.path.splitext(new_name)
            while (part, class_folder, new_name.lower()) in used_paths:
                new_name = f"{stem}_{number}{extension}"
                number += 1
            used_paths.add((part, class_folder, new_name.lower()))
            if new_name != file_name:
                renamed += 1

            destination = target_folder / part / class_folder / new_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as file:
                shutil.copyfileobj(source, file)
    print(f"PlantDoc unpacked ({renamed} photos got a Windows-safe name)")


# ----------------------------------------------------------------------------
# The three datasets
# ----------------------------------------------------------------------------
def download_plantvillage():
    if (PLANTVILLAGE_DIR / "raw" / "color").exists():
        print("PlantVillage already downloaded")
        return
    delete_folder(PLANTVILLAGE_DIR)
    # The repository is big, so we only check out the colour photos and the leaf-id file ("sparse checkout")
    run(f"git clone --filter=blob:none --no-checkout {PLANTVILLAGE_URL} {PLANTVILLAGE_DIR.name}", DATASETS_DIR)
    run('git sparse-checkout set --no-cone "/raw/color/*" "/leaf_grouping/leaf-map.json"', PLANTVILLAGE_DIR)
    run("git checkout HEAD", PLANTVILLAGE_DIR)


def download_plantdoc():
    if (PLANTDOC_DIR / "train").exists() and (PLANTDOC_DIR / "test").exists():
        print("PlantDoc already downloaded")
        return
    delete_folder(PLANTDOC_DIR)                          # leftovers of a failed `git clone`
    zip_path = DATASETS_DIR / "plantdoc.zip"
    download_file(PLANTDOC_ZIP_URL, zip_path)
    unzip_plantdoc(zip_path, PLANTDOC_DIR)
    zip_path.unlink()                                    # the ZIP is not needed any more


def download_rice():
    if RICE_DIR.exists() and any(RICE_DIR.rglob("*.jpg")):
        print("Rice photos already downloaded")
        return
    delete_folder(RICE_DIR)
    run(f"git clone --depth 1 {RICE_URL} {RICE_DIR.name}", DATASETS_DIR)


def download_all():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    download_plantvillage()
    download_plantdoc()
    download_rice()
    print("Datasets are in", DATASETS_DIR)
