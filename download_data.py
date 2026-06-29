import os
import zipfile
import gdown

FILE_ID = "1A8AWs0njeLnnwEBRuf_G7VUk_YGZ2GIA"

ZIP_NAME = "data-easy-actor-zip.zip"
DATA_FOLDER = "data"

if os.path.exists(DATA_FOLDER):
    print("Dataset already exists.")
else:
    print("Downloading dataset...")

    gdown.download(
        id=FILE_ID,
        output=ZIP_NAME,
        quiet=False
    )

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(".")

    os.remove(ZIP_NAME)

    print("Done.")
