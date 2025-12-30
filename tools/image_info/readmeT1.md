# Image Info Tool

Reports:
- width/height
- sharpness (Laplacian variance + Tenengrad)
- estimated noise (single-image)

## Setup (optional)
Use your radar project's venv or create a new one.

### Install deps into current venv
pip install -r tools/image_info/requirementsT1.txt

## Run
python tools/image_info/image_report.py path/to/image.jpg --pretty