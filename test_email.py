from alerts.notifier import EmailNotifier
from dotenv import load_dotenv
import os

load_dotenv()

print("Testing email notification system...")
print(f"Sender: {os.getenv('SENDER_EMAIL')}")
print(f"Recipient: {os.getenv('RECIPIENT_EMAIL')}")

notifier = EmailNotifier(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    sender_email=os.getenv('SENDER_EMAIL'),
    sender_password=os.getenv('SENDER_PASSWORD'),
    recipient_email=os.getenv('RECIPIENT_EMAIL')
)

# Test email
if notifier.send_zone_violation(1, "Test Zone", "John Doe"):
    print("\n✓ Test email sent successfully! Check your inbox.")
else:
    print("\n✗ Failed to send email. Check your .env credentials.")
