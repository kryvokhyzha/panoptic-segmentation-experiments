poetry_get_lock:
	poetry lock
poetry_update_deps:
	poetry update
poetry_update_self:
	poetry self update
poetry_show_deps:
	poetry show
poetry_show_deps_tree:
	poetry show --tree
poetry_build:
	poetry build

install_deps:
	poetry lock && poetry install --no-root && mim install mmengine && mim install "mmcv>=2.0.0"

download_cityscapes:
	python scripts/download_cityscapes.py \
			--file_id=1PMPMfEKWK0kvwadQAtvcrEY1SDTYvRgt && \
	mkdir -p data && \
	unzip -qq -o cityscapes.zip -d ./data/cityscapes && \
	rm cityscapes.zip && \
	python scripts/download_cityscapes.py \
			--file_id=1eUn338xKhhmJ6ykfWu0_CAwVMLZ7aFx1 && \
	mkdir -p data && \
	unzip -qq -o cityscapes.zip -d ./data/cityscapes && \
	rm cityscapes.zip

convert_cityscapes_to_coco_1:
	git clone https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion.git && \
	mkdir -p logs && \
	cd cityscapes-to-coco-conversion && \
	python -m venv convertor_venv && \
	source convertor_venv/bin/activate && \
	pip install -qq -r requirements.txt && \
	python main.py \
        --dataset cityscapes \
        --datadir ../data/cityscapes \
        --outdir ../data/cityscapes/annotations \
        >> ../logs/convert_cityscapes_to_coco_1.log && \
	deactivate && \
	cd .. && \
	rm -rf cityscapes-to-coco-conversion
convert_cityscapes_to_coco_2:
	mkdir -p logs && \
	git clone https://github.com/open-mmlab/mmdetection.git && \
	python mmdetection/tools/dataset_converters/cityscapes.py \
		./data/cityscapes \
		--nproc 8 \
		--out-dir ./data/cityscapes/annotations >> logs/convert_cityscapes_to_coco_2.log && \
	rm -rf mmdetection
convert_cityscapes_to_coco_3:
	mkdir -p data/cityscapes/annotations && \
	python scripts/convert_cityscapes_to_coco_panoptic.py \
			--gt_folder_path=data/cityscapes/gtFine/test/ \
			--gt_output_folder_path=data/cityscapes/gtFine/cityscapes_panoptic_test/ \
			--gt_output_annotations_file_path=data/cityscapes/annotations/cityscapes_panoptic_test.json \
			--img_folder_path=data/cityscapes/leftImg8bit/test \
			--img_output_folder_path=data/cityscapes/leftImg8bit/cityscapes_panoptic_test \
			--n_jobs=6 && \
	python scripts/convert_cityscapes_to_coco_panoptic.py \
			--gt_folder_path=data/cityscapes/gtFine/val/ \
			--gt_output_folder_path=data/cityscapes/gtFine/cityscapes_panoptic_val/ \
			--gt_output_annotations_file_path=data/cityscapes/annotations/cityscapes_panoptic_val.json \
			--img_folder_path=data/cityscapes/leftImg8bit/val \
			--img_output_folder_path=data/cityscapes/leftImg8bit/cityscapes_panoptic_val \
			--n_jobs=6 && \
	python scripts/convert_cityscapes_to_coco_panoptic.py \
			--gt_folder_path=data/cityscapes/gtFine/train/ \
			--gt_output_folder_path=data/cityscapes/gtFine/cityscapes_panoptic_train/ \
			--gt_output_annotations_file_path=data/cityscapes/annotations/cityscapes_panoptic_train.json \
			--img_folder_path=data/cityscapes/leftImg8bit/train \
			--img_output_folder_path=data/cityscapes/leftImg8bit/cityscapes_panoptic_train \
			--n_jobs=6

show_coco_annotations_gt_example: data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json
	head -c 250 data/cityscapes/annotations/cityscapes_panoptic_val.json

download_mask2former:
	mim download mmdet --config mask2former_r50_8xb2-lsj-50e_coco-panoptic --dest models/

run_inference_example:
	python scripts/image_demo.py \
		data/demo.jpg \
        models/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py \
        --weights models/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth \
        --device cpu
infer_one_file:
	python scripts/inference_one_image.py \
			--image_path=data/cityscapes/leftImg8bit/train//hamburg/hamburg_000000_000629_leftImg8bit.png
run_test:
	python scripts/test.py \
			./work_dirs/mask2former_r50_8xb2-lsj-50e_cityscapes-panoptic_train/mask2former_r50_8xb2-lsj-50e_cityscapes-panoptic_train.py \
			./work_dirs/mask2former_r50_8xb2-lsj-50e_cityscapes-panoptic_train/iter_1000.pth
run_train:
	python scripts/train.py models/mask2former_r50_8xb2-lsj-50e_cityscapes-panoptic_train.py

pre_commit_install: .pre-commit-config.yaml
	pre-commit install
pre_commit_run: .pre-commit-config.yaml
	pre-commit run --all-files
pre_commit_rm_hooks:
	pre-commit --uninstall-hooks

nvsmi0:
	watch -n 0.1 nvidia-smi -i 0
show_logfile:
	tail -f <path_to_file>
