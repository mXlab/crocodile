from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from apiclient import http
from pathlib import Path
import hashlib


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def check_integrity(path: Path, md5=None) -> bool:
    if not path.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)


def check_md5(path: Path, md5, **kwargs) -> bool:
    return md5 == calculate_md5(path, **kwargs)


def calculate_md5(path: Path, chunk_size: int = 1024*1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


class GoogleDrive:
    def __init__(self, drive=None):
        self.drive = drive

    def get_file_metadata(self, file_id: str):
        query = self.drive.files().get(fileId=file_id)
        metadata = query.execute()
        return metadata

    def download_file(self, file_id: str, path: Path, md5=None):
        if check_integrity(path, md5):
            return

        path.parent.mkdir(exist_ok=True)
        request = self.drive.files().get_media(fileId=file_id)
        with open(path, "wb") as f:
            downloader = http.MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Downloading %s %d%%." %
                      (path, int(status.progress() * 100)))

    @staticmethod
    def connect_to_drive(token: Path, scopes=SCOPES):
        creds = None

        if token.exists():
            creds = Credentials.from_authorized_user_file(token, scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                credentials_path = Path(input(
                    "Credentials file (default = './google-credentials.json'):") or "./google-credentials.json")
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, scopes)
                creds = flow.run_local_server(port=0)

            with open(token, 'w') as _token:
                _token.write(creds.to_json())

        drive = build('drive', 'v3', credentials=creds)
        return GoogleDrive(drive)
