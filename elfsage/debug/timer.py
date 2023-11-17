import time


class Timer:
    def __init__(self, logger=None, unit='ms'):
        assert unit in ['ms', 's']

        if logger is not None:
            self._log = logger
        else:
            self._init_logger()

        self._unit = unit

        self._message_mask = '{stage_time: 10.3f}{unit} ({full_time: 10.3f}{unit}) | {event}'
        self._init_t = time.time()
        self._t = time.time()

    def __call__(self, *args, **kwargs):
        if len(args)>0:
            event = args[0]
        else:
            event = 'Run'

        stage_time = time.time() - self._t
        full_time = time.time() - self._init_t

        if self._unit == 'ms':
            stage_time *= 1000
            full_time *= 1000

        message = self._message_mask.format(
            event=event,
            stage_time=stage_time,
            full_time=full_time,
            unit=self._unit
        )
        self._log.debug(message)
        self._t = time.time()

    def _init_logger(self):
        import sys
        import logging
        import uuid

        self._log = logging.getLogger(str(uuid.uuid4()))
        self._log.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
        handler.setFormatter(formatter)
        self._log.addHandler(handler)

