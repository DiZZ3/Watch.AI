# notifications.py
from twilio.rest import Client

class Notifier:
    def __init__(self, account_sid, auth_token):
        self.client = Client(account_sid, auth_token)
        self.from_number = 'your_twilio_number'
        self.to_number = 'your_phone_number'

    def send_alert(self, message):
        self.client.messages.create(
            body=message,
            from_=self.from_number,
            to=self.to_number
        )