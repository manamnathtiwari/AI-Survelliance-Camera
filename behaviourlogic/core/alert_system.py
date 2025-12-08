import telebot
import cv2
import time
import os
import threading

class AlertSystem:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
        self.bot = None
        self.last_alert_time = 0
        self.alert_cooldown = 60  # Seconds between alerts
        
        if self.bot_token and self.bot_token != "YOUR_BOT_TOKEN_HERE":
            try:
                self.bot = telebot.TeleBot(self.bot_token)
                print("AlertSystem: Telegram Bot initialized.")
            except Exception as e:
                print(f"AlertSystem: Failed to initialize Telegram Bot: {e}")
        else:
            print("AlertSystem: Warning - No valid Bot Token provided. Alerts will be simulated.")

    def send_alert(self, frame, message):
        """
        Sends an alert with an image to Telegram.
        Runs in a separate thread to avoid blocking the main video loop.
        """
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            print(f"AlertSystem: Cooldown active. Skipping alert: {message}")
            return

        self.last_alert_time = current_time
        threading.Thread(target=self._send_telegram_task, args=(frame, message)).start()

    def _send_telegram_task(self, frame, message):
        print(f"AlertSystem: Sending alert - {message}")
        if not self.bot or not self.chat_id:
            print("AlertSystem: Bot not configured. Alert skipped.")
            return

        try:
            # Encode frame to memory buffer instead of saving to disk
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                self.bot.send_photo(self.chat_id, encoded_img.tobytes(), caption=f"🚨 ALERT! 🚨\n{message}")
                self.bot.send_message(self.chat_id, f"{message}\nPlease take necessary precautions immediately!")
                print("AlertSystem: Alert sent successfully.")
            else:
                print("AlertSystem: Failed to encode image.")
        except Exception as e:
            print(f"AlertSystem: Error sending Telegram alert: {e}")
