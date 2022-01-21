import os, sys

MBUNA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MBUNA_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'datasets')

VERSION = '0.0.1'
AUTHOR = 'Tucker Lancaster'
VERSION_LONG = 'mbuna v{VERSION}, {AUTHOR}'

# package metadata
# _META = metadata("mbuna")
# NAME = _META["name"]
# VERSION = _META["version"]
# DESCRIPTION = _META["summary"]
# AUTHOR = _META["author"]
# AUTHOR_EMAIL = _META["author-email"]
# URL = _META["home-page"]
# LICENSE = _META["license"]
# VERSION_LONG = "FiftyOne v%s, %s" % (VERSION, AUTHOR)
