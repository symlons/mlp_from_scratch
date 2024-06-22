#!/usr/bin/env python
import requests
import gzip
import os
import hashlib
import numpy


def fetch(url):
    fp = os.path.join('/tmp', hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            dat = f.read()
    else:
        with open(fp, 'wb') as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()


def mnist():
    urls = ['https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz']
    data = []
    data.extend((fetch(urls[0])[0x10:], fetch(urls[1])[
                8:], fetch(urls[2])[0x10:], fetch(urls[3])[8:]))
    return data
