"""Library for Google Drive API."""
from typing import Any, Dict, Optional, Protocol, TypedDict
from pathlib import Path
import hashlib
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from apiclient import http

SCOPES = ["https://www.googleapis.com/auth/drive"]
default_credentials_path = "./crocodile-333216-61f547fcafc2.json"


def check_integrity(path: Path, md5=None) -> bool:
    """Check if the file at path has the expected md5."""
    if not path.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)


def check_md5(path: Path, md5, **kwargs) -> bool:
    """Check if the file at path has the expected md5."""
    return md5 == calculate_md5(path, **kwargs)


def calculate_md5(path: Path, chunk_size: int = 1024 * 1024):
    """Calculate the md5 for a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


class FileObject(TypedDict):
    id: str


class DriveQuery(Protocol):
    """Protocol for Google Drive Query."""

    def execute(self) -> FileObject:
        """Execute the query."""
        ...


class DriveFiles(Protocol):
    """Protocol for Google Drive Files API."""

    def get(self, fileId: str) -> DriveQuery:
        """Get a file."""
        ...

    def get_media(self, fileId: str) -> DriveQuery:
        """Get a file."""
        ...

    def create(
        self,
        body: Dict[str, Any],
        fields: str,
        media_body: Optional[http.MediaFileUpload] = None,
    ) -> DriveQuery:
        """Create a file."""
        ...


class Drive(Protocol):
    """Protocol for Google Drive API."""

    def files(self) -> DriveFiles:
        """Files API."""
        ...


class GoogleDrive:
    """Class to interact with Google Drive API."""

    def __init__(self, drive: Drive):
        self.drive = drive

    def get_file_metadata(self, file_id: str):
        """Get metadata for a file."""
        query = self.drive.files().get(fileId=file_id)
        metadata = query.execute()
        return metadata

    def download_file(self, file_id: str, path: Path, md5=None):
        """Download a file from Google Drive."""
        if check_integrity(path, md5):
            return

        path.parent.mkdir(exist_ok=True)
        request = self.drive.files().get_media(fileId=file_id)
        with open(path, "wb") as f:
            downloader = http.MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Downloading {path} {status.progress():.0%}.")

    def create_folder(self, name: str, folder_id: str):
        """Create a folder."""
        print(f"Creating folder {name}")
        folder_metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [folder_id],
        }
        folder = self.drive.files().create(body=folder_metadata, fields="id").execute()
        return folder.get("id")

    def upload_folder(self, folder_id: str, path: Path, name=None):
        """Upload a folder to Google Drive."""
        if not path.is_dir():
            raise FileNotFoundError(f"Path {path} is not a valid directory.")

        if name is None:
            name = path.stem

        folder_id = self.create_folder(name, folder_id)

        if name is None:
            name = path.stem

        folder_id = self.create_folder(name, folder_id)

        for child in path.iterdir():
            if child.is_dir():
                folder_id = self.create_folder(child.stem, folder_id)
                self.upload_folder(folder_id, child)
            if child.is_file():
                self.upload_file(folder_id, child)

    def upload_file(self, folder_id: str, path: Path):
        """Upload a file to Google Drive."""
        if not path.is_file():
            raise FileNotFoundError(f"Path {path} is not a valid file.")

        print(f"Uploading file {path}")
        file_metadata = {"name": path.name, "parents": [folder_id]}

        media = http.MediaFileUpload(path, resumable=True)
        file = (
            self.drive.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        return file.get("id")

    @staticmethod
    def connect_to_drive(scopes=SCOPES):
        credentials_path = Path(
            input(f"Credentials file (default = '{default_credentials_path}'):")
            or default_credentials_path
        )

        if not credentials_path.is_file():
            raise FileNotFoundError(
                f"Credentials file not found at {credentials_path}. The file can be downloaded here: https://drive.google.com/file/d/1vB3lQVu7EVKSbSjctY3G4_DlkzjoHjY3/view?usp=drive_link. If you don't have access, please request access to the file by sending an email to berard.hugo@gmail.com"
            )

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        credentials = credentials.with_scopes(scopes)

        drive = build("drive", "v3", credentials=credentials)
        return GoogleDrive(drive)
