import logging
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import re
import sys
LOG_PATH='E:\\project\\2021_Project\\TornadoWeb\\web\\logs\\'
sys.path.append('E:\\project\\2021_Project\\TornadoWeb\web\\xmodule\\')
sys.path.append('E:\\project\\2021_Project\\TornadoWeb\web\\log\\')
scriptPath = 'E:\\project\\2021_Project\\TornadoWeb\web\\customize\\'
sys.path.append(scriptPath)
def log_init():
    log_fmt = '%(asctime)s\\tFile \\"%(filename)s\\",line %(lineno)s\\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)
    log_file_handler = TimedRotatingFileHandler(filename=LOG_PATH+"thread_.log", when="D", interval=1, backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    log_file_handler.setFormatter(formatter)
    log_file_handler.setLevel(logging.DEBUG)
    log = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        # datefmt='%a, %d %b %Y %H:%M:%S',
                        # filename='/home/URL/client/test_log.log',
                        filemode='a')
    log.addHandler(log_file_handler)