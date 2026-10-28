
import logging
import os


class CustomLogger:
    def __init__(self):
        # Vercel serverless functions can write to /tmp.
        # Locally, keep logs in the project's logs directory.
        if os.getenv("VERCEL"):
            self.log_dir = "/tmp/logs"
        else:
            self.log_dir = "logs"

        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger("CustomGPT")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers when the app is initialized again.
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            log_file = os.path.join(self.log_dir, "app.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
