import logging
import os

class CaptionLogger:
    def __init__(self, log_file):
        self.logger = logging.getLogger("CaptionLogger")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add the handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def log(self, message):
        self.logger.info(message)
