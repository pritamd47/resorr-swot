import geopandas as gpd
from pathlib import Path
import xarray as xr
import rioxarray as rxr
import numpy as np
import networkx as nx
import geonetworkx as gnx


def generate_network(
        flow_dir_fn, 
        stations_fn, 
        save_dir=None, 
        dist_proj=None, 
        elevation_fn=None
    ) -> (gpd.GeoDataFrame, gpd.GeoDataFrame): 
    """
    Generate a reservoir network using a flow direction file and reservoir locations.

    This function creates a directed graph representing the reservoir network based on the flow direction raster and reservoir location data. The function identifies the flow paths between reservoirs, optionally calculates distances, and can include elevation data.

    Args:
        flow_dir_fn (str or Path): Path to the flow direction file. 
            The flow direction raster should use the following pixel values to indicate flow direction:
            - 1: North (N)
            - 2: Northeast (NE)
            - 3: East (E)
            - 4: Southeast (SE)
            - 5: South (S)
            - 6: Southwest (SW)
            - 7: West (W)
            - 8: Northwest (NW)
        stations_fn (str or Path): Path to the file containing reservoir locations. 
            This file should include the following columns:
            - 'name': The name of the reservoir.
            - 'lon': The longitude of the reservoir.
            - 'lat': The latitude of the reservoir.
        save_dir (str or Path, optional): Directory where the output files (edges and nodes) will be saved. Defaults to None.
        dist_proj (str, optional): Projection used for calculating distances between reservoirs. If provided, distances will be calculated in this projection. Defaults to None.
        elevation_fn (str or Path, optional): Path to an elevation raster file. If provided, elevation data will be added to the nodes. Defaults to None.

    Returns:
        tuple of gpd.GeoDataFrame: A tuple containing:
            - `edges` (gpd.GeoDataFrame): The GeoDataFrame representing the edges (flow paths) in the network.
            - `nodes` (gpd.GeoDataFrame): The GeoDataFrame representing the nodes (reservoirs) in the network.

    Raises:
        AssertionError: If the provided files do not exist.
    """

    # check if files exist
    flow_dir_fn = Path(flow_dir_fn)
    assert flow_dir_fn.exists()

    stations_fn = Path(stations_fn)
    assert stations_fn.exists()

    if save_dir:
        save_dir = Path(save_dir)
        if not save_dir.exists():
            print(f"passed save_dir does not exist, creating {save_dir}")
            save_dir.mkdir(parents=True)
    
    if elevation_fn:
        elevation_fn = Path(elevation_fn)
        assert elevation_fn.exists()

    fdr = rxr.open_rasterio(flow_dir_fn, masked=True)
    band = fdr.sel(band=1)

    band_vicfmt = band

    reservoirs = gpd.read_file(stations_fn)
    reservoirs['geometry'] = gpd.points_from_xy(reservoirs['lon'], reservoirs['lat'])
    reservoirs.set_crs('epsg:4326', inplace=True)

    reservoir_location_raster = xr.full_like(band_vicfmt, np.nan)
    for resid, row in reservoirs.iterrows():
        reslat = float(row.lat)
        reslon = float(row.lon)

        rast_lat = reservoir_location_raster.indexes['y'].get_indexer([reslat], method="nearest")[0]
        rast_lon = reservoir_location_raster.indexes['x'].get_indexer([reslon], method="nearest")[0]

        reservoir_location_raster[rast_lat, rast_lon] = resid

    # convert all points to nodes. Use index value to identify
    G = gnx.GeoDiGraph()
    G.add_nodes_from(reservoirs.index)

    operations = {
        1: [-1, 0],  # N
        2: [-1, 1],  # NE
        3: [0, 1],   # E
        4: [1, 1],   # SE
        5: [1, 0],   # S
        6: [1, -1],  # SW
        7: [0, -1],  # W
        8: [-1, -1], # NW
    }

    for node in G.nodes:
        resdata = reservoirs[reservoirs.index==node]
        
        x = float(resdata['lon'].values[0])
        y = float(resdata['lat'].values[0])

        idxx = band_vicfmt.indexes['x'].get_indexer([x], method="nearest")[0]
        idxy = band_vicfmt.indexes['y'].get_indexer([y], method="nearest")[0]
        
        # travel downstream until another node, np.nan or out-of-bounds is found, or if travelling in a loop

        visited = [(idxx, idxy)]
        current_pix = band_vicfmt.isel(x=idxx, y=idxy)

        attrs_n = {
            node: {
                'x': reservoirs['geometry'][node].x,
                'y': reservoirs['geometry'][node].y,
                'name': reservoirs['name'][node]
            }
        }
        nx.set_node_attributes(G, attrs_n)

        if not np.isnan(current_pix):
            END = False
            while not END:
                op = operations[int(current_pix)]
                new_idxy, new_idxx = np.array((idxy, idxx)) + np.array(op)
                idxy, idxx = new_idxy, new_idxx
                
                if (new_idxx, new_idxy) in visited:
                    # In a loop, exit
                    END=True
                    break
                else:
                    visited.append((new_idxx, new_idxy))

                current_pix = band_vicfmt.isel(x=new_idxx, y=new_idxy)
                if np.isnan(current_pix):
                    # NaN value found, exit loop
                    END=True
                    break

                try:
                    any_reservoir = reservoir_location_raster.isel(x=new_idxx, y=new_idxy)
                    if not np.isnan(any_reservoir):
                        # another reservoir found
                        G.add_edge(node, int(any_reservoir))
                        if dist_proj:
                            attrs_e = {
                                (node, int(any_reservoir)): {
                                    'length': reservoirs.to_crs(dist_proj)['geometry'][node].distance(reservoirs.to_crs(dist_proj)['geometry'][int(any_reservoir)])
                                }
                            }
                            nx.set_edge_attributes(G, attrs_e)
                        END = True
                        break
                except IndexError:
                    # Reached end
                    END=True

    G_gdf = gpd.GeoDataFrame(gnx.graph_edges_to_gdf(G))
    G_gdf_pts = gpd.GeoDataFrame(gnx.graph_nodes_to_gdf(G))

    # add elevation data to nodes
    if elevation_fn:
        elev = rxr.open_rasterio(elevation_fn, chunks='auto')

        G_gdf_pts['elevation'] = G_gdf_pts[['x', 'y']].apply(lambda row: float(elev.sel(x=row.x, y=row.y, method='nearest')), axis=1)
        G_gdf_pts.head()

    if save_dir:
        pts_save_fn = Path(save_dir) / 'rivreg_network_pts.shp'
        edges_save_fn = Path(save_dir) / 'rivreg_network.shp'
        
        G_gdf_pts.to_file(pts_save_fn)
        G_gdf.to_file(edges_save_fn)

    return G