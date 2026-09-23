from pathlib import Path
from PIL import Image
import pytest

DATA = Path(__file__).parents[1] / "data" / "animals"

def test_data_folder_exists():
    assert DATA.is_dir()

def test_has_at_least_two_valid_class_folders():
    folders = [p for p in DATA.iterdir() if p.is_dir()]
    assert len(folders) >= 2
    assert all(p.name == p.name.lower() and " " not in p.name for p in folders)

def test_no_loose_images():
    assert not any(p.is_file() for p in DATA.iterdir())

@pytest.mark.parametrize("animal_dir", sorted([p for p in DATA.iterdir() if p.is_dir()]) if DATA.is_dir() else [])
def test_each_animal_has_readable_images(animal_dir):
    images = list(animal_dir.glob("*.jpg"))
    assert len(images) >= 10
    for image in images:
        with Image.open(image) as opened:
            opened.verify()