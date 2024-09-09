from abc import abstractmethod

# Gmail APIのスコープを設定
SCOPES = ['https://www.googleapis.com/auth/gmail.send']


class Messenger:

    def __init__(self):
        pass

    @abstractmethod
    def send_message(self, subject: str, body: str):
        pass


''''
if __name__ == '__main__':
    mail = Messenger()
    mail.send_mail("未来の自分へ", "これを読んでいるということは、私のメールは正しく届いたんだね。<br>本当によかった")
'''
