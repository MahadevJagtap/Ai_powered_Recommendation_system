import ssl
import certifi
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io

DATA_DIR = Path("data/deployment")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

print("Downloading MovieLens latest-small...")
ssl_context = ssl.create_default_context(cafile=certifi.where())

with urlopen(URL, timeout=120, context=ssl_context) as response:
    archive = response.read()

print("Extracting dataset...")
with ZipFile(io.BytesIO(archive)) as zip_file:
    zip_file.extractall(DATA_DIR)

dataset_path = DATA_DIR / "ml-latest-small"
if not (dataset_path / "ratings.csv").is_file():
    raise RuntimeError("Dataset extraction failed: ratings.csv is missing")
if not (dataset_path / "movies.csv").is_file():
    raise RuntimeError("Dataset extraction failed: movies.csv is missing")

print(f"Dataset ready at {dataset_path}")
