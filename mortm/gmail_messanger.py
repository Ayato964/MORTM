import os
import base64

from mortm.messager import Messenger
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText


# Gmail APIのスコープを設定
SCOPES = ['https://www.googleapis.com/auth/gmail.send']


class GmailMessanger(Messenger):

    def __init__(self, token_file: str, client_secret_file: str, send_address: str, step_by_message_count=100):
        super().__init__(step_by_message_count=step_by_message_count)
        self.token_file = token_file
        self.client_secret_file = client_secret_file
        self.send_address = send_address
        self.creds = self.authenticate_gmail()

    def authenticate_gmail(self):
        creds = None
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, SCOPES)
                creds = flow.run_local_server(port=0)
            # 認証情報を保存
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        return creds


    def create_message(self, to, subject, message_text):
        message = MIMEText(message_text)
        message['to'] = to
        message['subject'] = subject
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}


    def _send_email(self, service, user_id, message):
        try:
            message = service.users().messages().send(userId=user_id, body=message).execute()
            print(f'Message Id: {message["id"]}')
            return message
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None


    def send_message(self, subject: str, body: str):
        # 認証を実行してGmail APIサービスを取得
        service = build('gmail', 'v1', credentials=self.creds)

        message = self.create_message(self.send_address, subject, body)

        # メールを送信
        self._send_email(service, 'me', message)
