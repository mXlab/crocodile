from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from apiclient import http
from pathlib import Path
import hashlib


SCOPES = ["https://www.googleapis.com/auth/drive"]


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

    def create_folder(self, name, folder_id):
        print("Creating folder %s" % name)
        folder_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [folder_id]
        }
        folder = self.drive.files().create(body=folder_metadata,
                                           fields='id').execute()
        return folder.get('id')

    def upload_folder(self, folder_id, path):
        if not path.is_dir():
            raise("Path %s is not a vail directory." % path)

        for child in path.iterdir():
            if child.is_dir():
                folder_id = self.create_folder(child.stem, folder_id)
                self.upload_folder(folder_id, child)
            if child.is_file():
                self.upload_file(folder_id, child)

    def upload_file(self, folder_id, path):
        if not path.is_file():
            raise("Path %s is not a valid file." % path)

        print("Uploading file %s." % path)
        file_metadata = {
            'name': path.name,
            'parents': [folder_id]
        }

        media = http.MediaFileUpload(path,
                                     resumable=True)
        file = self.drive.files().create(body=file_metadata,
                                         media_body=media,
                                         fields='id').execute()
        return file.get('id')

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
