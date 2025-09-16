import papermill as pm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl

import geopandas as gpd
from pathlib import Path

from dask.distributed import LocalCluster, Client

import warnings

warnings.filterwarnings('ignore')


def process(resname, PARAMS, notebook_path):
    notebook_path = Path(notebook_path)
    notebook_name = notebook_path.stem
    # dst_notebook_path = notebook_path.parent / 'papermill' / notebook_name / f"{reservoir}.ipynb"
    NODE = PARAMS.get('NODE')
    dst_notebook_path = notebook_path.parent / 'papermill' / notebook_name / f"{NODE}.ipynb"

    if not dst_notebook_path.parent.exists():
        print("Creating directory to save notebooks ran through papermill: ", str(dst_notebook_path.parent))
        dst_notebook_path.parent.mkdir()

    print(f"processing {resname}")
    try:
        pm.execute_notebook(
            notebook_path,
            dst_notebook_path,
            parameters=PARAMS
        )
        return True
    except Exception as e:
        # print(f'Something went wrong, {reservoir}: {resname}')
        print(e)
        return False

def main(client=None):
    NOTEBOOK = "/tiger1/pdas47/resorr-swot/notebooks/resorr/02-Regulation.ipynb"

    RESERVOIR_BOUNDARIES = gpd.read_file(Path('/tiger1/pdas47/resorr-swot/data/cumberland-reservoirs/03-cumberland-reservoirs.geojson'))
    RESERVOIR_WITH_UPSTREAM = [0, 2, 4, 8]

    # selected_reservoirs = RESERVOIR_BOUNDARIES['id'].tolist()
    selected_reservoirs = RESERVOIR_WITH_UPSTREAM
    res_names_dict = RESERVOIR_BOUNDARIES[['id', 'name']].set_index('id').to_dict()['name']
    res_names = [res_names_dict[res] for res in selected_reservoirs]
    
    if client is None:
        for reservoir, resname in zip(selected_reservoirs, res_names):
            PARAMS = dict(
                NODE=reservoir,
            )
            process(resname, PARAMS, NOTEBOOK)
    else:
        futures = client.map(process, res_names, [dict(NODE=reservoir) for reservoir in selected_reservoirs], [NOTEBOOK]*len(selected_reservoirs))
        results = client.gather(futures)
        print(results)
        print(f'Passed: {results.count(True)}/{len(results)}')

if __name__ == '__main__':
    clutser = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(clutser)
    print(client.dashboard_link)

    main(client)