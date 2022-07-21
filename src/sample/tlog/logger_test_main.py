from src.sample.tlog.logger import getLogger
logger = getLogger(debug=False, filename="./data/sample.log", add_stream_handler=True)


from src.sample.tlog.logger_test_src import logtest_func

logger.info("info message")
logger.debug("debug message")
logger.error("error message")
logtest_func()
