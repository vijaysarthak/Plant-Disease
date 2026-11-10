"""
dataset_split.py — makes the lists of images and splits them FAIRLY.

The problem: PlantVillage photographed every leaf 5-7 times. If photos of the SAME leaf land in
both training and testing, the test score is "too good" because the model just remembers the leaf.
So we split by LEAF, not by photo:

  1. `leaf-map.json` (an official PlantVillage file) tells us the leaf of ~76% of the photos.
  2. For the other photos we use a "perceptual hash" (pHash): a 64-bit fingerprint of how a photo
     looks. Two photos whose fingerprints differ in only a few bits look almost the same.
  3. Then whole leaves are divided: 70% train, 10% val, 10% cal, 10% test.

PlantDoc (field photos) has the opposite problem: some internet photos appear twice, a few even
with two DIFFERENT labels. Those are removed, and copies are kept together ("dup_group").

Every image is stored as a small dictionary ("row"), for example:
    {"path": ".../img1.jpg", "class_name": "Tomato___Late_blight", "label": 29, "leaf": "...", "split": "train"}

Used by: the notebook only (needs `imagehash`, which the app does not need).
"""
import json
import os
import random
from collections import defaultdict

import imagehash
from PIL import Image

from src.disease_info import CLASS_NAMES, PLANTDOC_TO_PLANTVILLAGE

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def list_images(folder):
    """All image file names in a folder, sorted (so every run gives the same order)."""
    names = []
    for name in os.listdir(folder):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            names.append(name)
    return sorted(names)


def photo_fingerprint(path):
    """The photo's 64-bit perceptual hash as a whole number (None if the file is broken)."""
    try:
        with Image.open(path) as image:
            return int(str(imagehash.phash(image.convert("RGB"))), 16)
    except Exception:
        return None


def group_similar_photos(fingerprints, max_different_bits):
    """
    Put photos whose fingerprints differ in at most `max_different_bits` bits into the same group.
    Returns one group number per photo.

    How (a "union-find"): every photo starts as its own group. `parent[i]` points to another photo
    of the same group; following the pointers ends at the group's "root" photo. Joining two groups
    = pointing one root at the other.
    """
    parent = list(range(len(fingerprints)))

    def find_root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            different_bits = bin(fingerprints[i] ^ fingerprints[j]).count("1")
            if different_bits <= max_different_bits:
                root_i = find_root(i)
                root_j = find_root(j)
                parent[root_i] = root_j

    groups = []
    for i in range(len(fingerprints)):
        groups.append(find_root(i))
    return groups


def leaf_map_key(file_name):
    """Turns a PlantVillage file name into the key used inside leaf-map.json."""
    key = file_name.replace("_final_masked", "")
    if "___" in key:
        key = key.split("___")[-1]
    key = key.split("copy")[0]
    for extension in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
        key = key.replace(extension, "")
    return key.strip().lower()


# ----------------------------------------------------------------------------
# PlantVillage (lab photos)
# ----------------------------------------------------------------------------
def find_plantvillage_images(plantvillage_dir, max_per_class=None):
    """One row per PlantVillage photo, with its leaf id from leaf-map.json (or None)."""
    leaf_map = json.load(open(plantvillage_dir / "leaf_grouping" / "leaf-map.json"))
    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_folder = plantvillage_dir / "raw" / "color" / class_name
        file_names = list_images(class_folder)
        if max_per_class is not None:
            file_names = file_names[:max_per_class]

        for file_name in file_names:
            leaf_id = None
            candidates = leaf_map.get(leaf_map_key(file_name))
            if candidates:
                if len(candidates) == 1:
                    leaf_id = candidates[0]
                else:
                    # the same key exists in several classes: take the one of this class
                    for candidate in candidates:
                        if class_name in candidate:
                            leaf_id = candidate
                            break
            rows.append({"path": str(class_folder / file_name), "class_name": class_name,
                         "label": label, "leaf": leaf_id})
    return rows


def fill_missing_leaf_ids(rows, max_different_bits=8):
    """Photos without an official leaf id: group look-alike photos of the same class with pHash."""
    for class_name in CLASS_NAMES:
        rows_without_id = []
        for row in rows:
            if row["class_name"] == class_name and row["leaf"] is None:
                rows_without_id.append(row)

        fingerprints = []
        good_rows = []
        for row in rows_without_id:
            fingerprint = photo_fingerprint(row["path"])
            if fingerprint is not None:
                fingerprints.append(fingerprint)
                good_rows.append(row)

        groups = group_similar_photos(fingerprints, max_different_bits)
        for row, group_number in zip(good_rows, groups):
            row["leaf"] = f"{class_name}::phash{group_number}"

    # a photo that could not be read has still no leaf id -> drop it
    return [row for row in rows if row["leaf"] is not None]


def split_by_leaf(rows, seed=42):
    """Inside every class: shuffle the LEAVES, then give 70% / 10% / 10% / 10% of the photos
    to train / val / cal / test. All photos of one leaf always go to the same part."""
    rng = random.Random(seed)
    for class_name in CLASS_NAMES:
        photos_of_leaf = defaultdict(list)
        for row in rows:
            if row["class_name"] == class_name:
                photos_of_leaf[row["leaf"]].append(row)

        leaves = sorted(photos_of_leaf)
        rng.shuffle(leaves)
        total_photos = sum(len(photos_of_leaf[leaf]) for leaf in leaves)

        photos_done = 0
        for leaf in leaves:
            fraction_done = photos_done / total_photos
            if fraction_done < 0.70:
                split = "train"
            elif fraction_done < 0.80:
                split = "val"
            elif fraction_done < 0.90:
                split = "cal"
            else:
                split = "test"
            for row in photos_of_leaf[leaf]:
                row["split"] = split
            photos_done += len(photos_of_leaf[leaf])
    return rows


def count_leaves_in_two_splits(rows):
    """Safety check: how many leaves appear in more than one split? (must be 0)"""
    splits_of_leaf = defaultdict(set)
    for row in rows:
        splits_of_leaf[row["leaf"]].add(row["split"])
    count = 0
    for splits in splits_of_leaf.values():
        if len(splits) > 1:
            count += 1
    return count


# ----------------------------------------------------------------------------
# PlantDoc (field photos) and rice (unknown crop)
# ----------------------------------------------------------------------------
def find_plantdoc_images(plantdoc_dir, max_per_class=None, max_different_bits=4):
    """One row per PlantDoc photo, renamed to the PlantVillage class.
    Adds 'dup_group' (copies of the same photo) and 'keep' (False = copies with conflicting labels)."""
    rows = []
    for part in ["train", "test"]:
        for folder_name in sorted(os.listdir(plantdoc_dir / part)):
            folder = plantdoc_dir / part / folder_name
            if not folder.is_dir():
                continue
            file_names = list_images(folder)
            if max_per_class is not None:
                file_names = file_names[:max_per_class]
            class_name = PLANTDOC_TO_PLANTVILLAGE[folder_name]
            for file_name in file_names:
                rows.append({"path": str(folder / file_name), "class_name": class_name,
                             "label": CLASS_NAMES.index(class_name), "plantdoc_folder": folder_name})

    # fingerprint every photo, drop broken files
    fingerprints = []
    good_rows = []
    for row in rows:
        fingerprint = photo_fingerprint(row["path"])
        if fingerprint is not None:
            fingerprints.append(fingerprint)
            good_rows.append(row)
    rows = good_rows

    # copies of the same photo get the same dup_group
    groups = group_similar_photos(fingerprints, max_different_bits)
    for row, group_number in zip(rows, groups):
        row["dup_group"] = f"dup{group_number}"

    # a dup_group whose copies carry different labels is "conflicting" -> do not use it
    labels_in_group = defaultdict(set)
    for row in rows:
        labels_in_group[row["dup_group"]].add(row["class_name"])
    for row in rows:
        row["keep"] = len(labels_in_group[row["dup_group"]]) == 1
    return rows


def find_rice_images(rice_dir):
    """All rice photos (any sub-folder). Label -1 = 'not one of our 38 classes'."""
    rows = []
    for path in sorted(rice_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"path": str(path), "class_name": "rice (unknown crop)", "label": -1})
    return rows
