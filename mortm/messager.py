from abc import abstractmethod


class Messenger:

    def __init__(self, step_by_message_count=100):
        self.step_by_message_count = step_by_message_count
        pass

    @abstractmethod
    def send_message(self, subject: str, body: str):
        pass


class _DefaultMessenger(Messenger):
    def __init__(self, step_by_message_count=100):
        super().__init__(step_by_message_count)

    def send_message(self, subject: str, body: str):
        #print(subject, body)
        pass
