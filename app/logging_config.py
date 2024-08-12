import logging
import logging.config
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'app.log'),
            'when': 'midnight',  # Rotate log file at midnight
            'backupCount': 7,  # Keep the last 7 log files
            'encoding': 'utf8'
        },
        'console': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
