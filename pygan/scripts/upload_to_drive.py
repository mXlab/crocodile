from pygan.utils.drive import GoogleDrive
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--token", default=Path("./token.json"), type=Path)
    parser.add_argument(
        "--id", default="1E9BtfIBMCWgFj7fwmwP7H5vV0HDvY0DG", type=str)

    args = parser.parse_args()
    drive = GoogleDrive.connect_to_drive(
        args.token, ["https://www.googleapis.com/auth/drive"])
    drive.upload_folder(args.id, args.input_dir)
