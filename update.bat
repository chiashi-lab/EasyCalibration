git pull origin main
CALL venv\Scripts\activate
python -m pip uninstall -y calibrator dataloader
python -m pip install git+https://github.com/chiashi-lab/Calibrator git+https://github.com/chiashi-lab/DataLoader git+https://github.com/chiashi-lab/Tooltip