from abc import abstractmethod


class Messenger:

    def __init__(self):
        pass

    @abstractmethod
    def send_message(self, subject: str, body: str):
        pass


class _DefaultMessenger(Messenger):
    def send_message(self, subject: str, body: str):
        print(subject, body)
        pass
