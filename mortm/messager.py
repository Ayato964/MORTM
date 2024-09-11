from abc import abstractmethod


class Messenger:

    def __init__(self):
        pass

    @abstractmethod
    def send_message(self, subject: str, body: str):
        pass
