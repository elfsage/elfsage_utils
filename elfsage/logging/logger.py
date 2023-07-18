import sys
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

# handler = TimedRotatingFileHandler('logs/HOSRecordProcessingService.log', when='midnight', interval=1, backupCount=0)
# handler.suffix = "%Y%m%d"
# handler.setLevel(logging.DEBUG)
# handler.setFormatter(formatter)
# logger.addHandler(handler)