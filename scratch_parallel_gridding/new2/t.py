#!/usr/bin/env python3

from dask import config as cfg

cfg.set({'distributed.scheduler.worker-ttl': None})
cfg.set({'logging.distributed': 'error'})

from dask_mpi import initialize
initialize()

from dask.distributed import Client

with Client() as client:
    print("dscsd")

