import sys

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

MIN_LEVEL = 30

def set_level(level):
    """
    Set logging threshold on current logger.
    """
    global MIN_LEVEL
    MIN_LEVEL = level

def debug(msg, *args):
    if MIN_LEVEL <= INFO:
        sys.stderr.write('%s: %s'%('DEBUG', msg % args))

def info(msg, *args):
    if MIN_LEVEL <= INFO:
        sys.stderr.write('%s: %s'%('INFO', msg % args))

def warn(msg, *args):
    if MIN_LEVEL <= WARN:
        sys.stderr.write('%s: %s'%('WARN', msg % args))

def error(msg, *args):
    if MIN_LEVEL <= ERROR:
        sys.stderr.write('%s: %s'%('ERROR', msg % args))