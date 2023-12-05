import fire
import gdown


def download_from_colab(file_id: str):
    output_file = "cityscapes.zip"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=True)


fire.Fire(download_from_colab)
