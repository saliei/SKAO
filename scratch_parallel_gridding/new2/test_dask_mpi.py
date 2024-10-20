#!/usr/bin/env python3

import dask_mpi
from dask_mpi import initialize
from distributed import LocalCluster, Client, progress
import dask

from dask import config as cfg

cfg.set({'distributed.scheduler.worker-ttl': None})
cfg.set({'logging.distributed': 'error'})
cfg.set({"distributed.worker.use-file-locking": False})

def square(x):
    return x ** 2


def main():

    is_client = initialize(nthreads=1, dashboard=False, exit=False)

    with LocalCluster(processes=True, threads_per_worker=1) as cluster:
        with Client(cluster, direct_to_workers=True) as client:
            if is_client:
                futures = client.map(square, range(1000))

                for key, obj in cluster.workers.items():
                    print(f"{key}: {obj}")

                progress(futures)
                results = client.gather(futures)

    print("is_client = ", is_client)

    if is_client:
        print(results[:20])
        dask_mpi.send_close_signal()


if __name__ == '__main__':
    main()
