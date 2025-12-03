from __future__ import annotations
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime


class EmailNotifier:
    """Simple email notification system for security alerts."""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 sender_email: str, sender_password: str,
                 recipient_email: str) -> None:
        """
        Args:
            smtp_server: 'smtp.gmail.com' for Gmail
            smtp_port: 587 for Gmail
            sender_email: Your Gmail address
            sender_password: Your Gmail app password
            recipient_email: Email to receive alerts
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        
        # Track last alert time to prevent spam
        self.last_alert_time = None
        self.cooldown_seconds = 60  # Wait 60 seconds between emails
    
    def can_send_alert(self) -> bool:
        """Check if we can send another alert (cooldown check)."""
        if self.last_alert_time is None:
            return True
        
        time_since_last = (datetime.now() - self.last_alert_time).total_seconds()
        return time_since_last >= self.cooldown_seconds
    
    def send_alert(self, subject: str, message: str) -> bool:
        """Send email alert.
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.can_send_alert():
            return False  # Skip to prevent spam
        
        try:
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            # Add timestamp and message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"Time: {timestamp}\n\n{message}\n\nSmart Surveillance System"
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure connection
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.last_alert_time = datetime.now()
            print(f"✓ Email sent: {subject}")
            return True
            
        except Exception as e:
            print(f"✗ Email failed: {str(e)}")
            return False
    
    def send_zone_violation(self, track_id: int, zone_name: str, 
                           person_name: Optional[str] = None) -> bool:
        """Send zone violation alert."""
        person = f" - Person: {person_name}" if person_name else ""
        subject = "🚨 Security Alert: Zone Violation"
        message = f"Restricted zone '{zone_name}' was entered!\n\nTrack ID: {track_id}{person}"
        return self.send_alert(subject, message)
