from src.sample.tlog.logger import getLogger

logger = getLogger()

def logtest_func():
    logger.info("func info message")
    logger.debug("func debug message")