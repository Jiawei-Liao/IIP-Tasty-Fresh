import glob
from PIL import Image
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
         
pattern = './*/dataset/*/*/*.png'

file_paths = glob.glob(pattern, recursive=True)
for file_path in file_paths:
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        print(f"Corrupted: {file_path} - {e}")
