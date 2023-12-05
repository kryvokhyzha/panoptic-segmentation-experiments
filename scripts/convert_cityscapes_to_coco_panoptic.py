import glob
import json
import os
import shutil

import fire
import numpy as np
import PIL.Image as Image
from joblib import Parallel, delayed
from panopticapi.utils import IdGenerator
from tqdm import tqdm


try:
    # set up path for cityscapes scripts
    # sys.path.append('./cityscapesScripts/')
    from cityscapesscripts.helpers.labels import id2label, labels
except Exception:
    raise Exception("Please load Cityscapes scripts from https://github.com/mcordts/cityscapesScripts")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def image_converter(f, categories_dict, out_folder):
    original_format = np.array(Image.open(f))

    file_name = f.split("/")[-1]
    image_id = "_".join(file_name.split("_")[:3])
    image_filename = "{}.png".format(image_id)
    segm_filename = "{}.png".format(image_id)

    # image entry, id for image is its filename without extension
    image_config = {
        "id": image_id,
        "width": original_format.shape[1],
        "height": original_format.shape[0],
        "file_name": image_filename,
    }

    pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
    id_generator = IdGenerator(categories_dict)

    l = np.unique(original_format)
    segm_info = []
    for el in l:
        if el < 1000:
            semantic_id = el
            is_crowd = 1
        else:
            semantic_id = el // 1000
            is_crowd = 0
        if semantic_id not in categories_dict:
            continue
        if categories_dict[semantic_id]["isthing"] == 0:
            is_crowd = 0
        mask = original_format == el
        segment_id, color = id_generator.get_id_and_color(semantic_id)
        pan_format[mask] = color

        area = np.sum(mask)  # segment area computation

        # bbox computation for a segment
        hor = np.sum(mask, axis=0)
        hor_idx = np.nonzero(hor)[0]
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        vert = np.sum(mask, axis=1)
        vert_idx = np.nonzero(vert)[0]
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1
        bbox = [x, y, width, height]

        segm_info.append(
            {
                "id": int(segment_id),
                "category_id": int(semantic_id),
                "area": area,
                "bbox": bbox,
                "iscrowd": is_crowd,
            }
        )

    annotation_config = {
        "image_id": image_id,
        "file_name": segm_filename,
        "segments_info": segm_info,
    }

    Image.fromarray(pan_format).save(os.path.join(out_folder, segm_filename))
    return image_config, annotation_config


def panoptic_converter(
    gt_folder_path: str,
    gt_output_folder_path: str,
    gt_output_annotations_file_path: str,
    img_folder_path: str,
    img_output_folder_path: str,
    n_jobs: int = 4,
    remove_folders: bool = False,
):
    if not os.path.isdir(gt_output_folder_path):
        print("Creating folder {} for panoptic segmentation GT PNGs".format(gt_output_folder_path))
        os.mkdir(gt_output_folder_path)

    if not os.path.isdir(img_output_folder_path):
        print("Creating folder {} for panoptic segmentation 8-bit PNGs".format(img_output_folder_path))
        os.mkdir(img_output_folder_path)

    categories = []
    for idx, el in tqdm(enumerate(labels), total=len(labels), desc="Adding categories"):
        if el.ignoreInEval:
            continue

        categories.append(
            {
                "id": el.id,
                "name": el.name,
                "color": el.color,
                "supercategory": el.category,
                "isthing": 1 if el.hasInstances else 0,
            }
        )

    categories_dict = {cat["id"]: cat for cat in categories}

    gt_file_list = sorted(glob.glob(os.path.join(gt_folder_path, "*/*_gtFine_instanceIds.png")))

    result = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(image_converter)(f, categories_dict, gt_output_folder_path)
        for f in tqdm(gt_file_list, total=len(gt_file_list), desc="Converting images")
    )
    images, annotations = list(zip(*result))

    d = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(gt_output_annotations_file_path, "w") as f:
        json.dump(d, f, cls=NpEncoder)

    if remove_folders:
        shutil.rmtree(gt_folder_path)

    img_file_list = sorted(glob.glob(os.path.join(img_folder_path, "*/*_leftImg8bit.png")))
    Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(shutil.copyfile)(
            f, os.path.join(img_output_folder_path, f"{'_'.join(f.split('/')[-1].split('_')[:3])}.png")
        )
        for f in tqdm(img_file_list, total=len(img_file_list), desc="Move 8-bit images")
    )

    if remove_folders:
        shutil.rmtree(img_folder_path)


fire.Fire(panoptic_converter)
