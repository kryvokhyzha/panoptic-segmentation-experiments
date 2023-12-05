from pathlib import Path
from typing import Optional

import fire
from mmdet.apis import DetInferencer
from rich.pretty import pprint


def infer_one_image(
    model_name: str = "mask2former_r50_8xb2-lsj-50e_coco-panoptic",
    image_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    path_to_root = Path(__file__).parent.parent

    if image_path is None:
        image_path = str(
            path_to_root.joinpath("data/cityscapes/leftImg8bit/train/krefeld/krefeld_000000_000108_leftImg8bit.png")
        )

    if output_path is None:
        output_path = str(path_to_root.joinpath("outputs"))

    # Initialize the DetInferencer
    inferencer = DetInferencer(model_name)

    # Perform inference
    inference_result = inferencer(image_path, return_vis=True, out_dir=output_path, return_datasamples=True)
    pprint(inference_result, max_length=4)


fire.Fire(infer_one_image)
