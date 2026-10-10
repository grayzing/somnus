import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
from shapely.ops import triangulate

# Ensure this matches your actual simulator filename
from simulator import Simulator 

def visualize_simulation_plotly_3d():
    print("Initializing simulator and fetching Downtown LA geometry...")
    num_base_stations = 7
    num_devices = 40
    sim = Simulator(num_base_stations=num_base_stations, num_devices=num_devices)
    
    # Reset and run a few steps to populate positions, channels, and associations
    sim.reset(seed=42)
    for step in range(5):
        sim.step(step)

    # Extract geometries and positions
    obstacles_gdf = sim.geometry_helper.obstacles_gdf.copy()
    
    # Use the correct getter for base stations as defined in base_station_manager_2.py
    bs_positions = sim.base_station_manager.get_all_positions()
    
    # Access device positions
    ue_positions = getattr(sim.device_manager, 'device_position_matrix', None)
    if ue_positions is None:
        ue_positions = sim.device_manager.get_all_positions()
        
    sector_parent = sim.sector_manager.sector_parent_base_station_vector
    association_matrix = sim.handover_manager.sector_device_association_matrix

    fig = go.Figure()

    # 1. Plot GIS Obstacles (Building Footprints) as Opaque 3D Meshes
    if obstacles_gdf is not None and not obstacles_gdf.empty:
        mesh_x, mesh_y, mesh_z = [], [], []
        mesh_i, mesh_j, mesh_k = [], [], []
        vertex_offset = 0
        
        for idx, row in obstacles_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            # Determine building height from OSM metadata (fallback to 15.0m)
            bldg_height = 15.0
            if 'height' in row and row['height']:
                try:
                    bldg_height = float(str(row['height']).replace('m', '').replace(',', '').split(';')[0])
                except (ValueError, TypeError):
                    pass
            elif 'building:levels' in row and row['building:levels']:
                try:
                    bldg_height = float(str(row['building:levels']).split(';')[0]) * 3.0
                except (ValueError, TypeError):
                    pass
            
            polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
            for poly in polys:
                # Extrude the exterior walls
                x_ext, y_ext = list(poly.exterior.xy[0]), list(poly.exterior.xy[1])
                for idx_p in range(len(x_ext) - 1):
                    x0, y0 = x_ext[idx_p], y_ext[idx_p]
                    x1, y1 = x_ext[idx_p+1], y_ext[idx_p+1]
                    
                    # 4 vertices for this wall segment
                    mesh_x.extend([x0, x1, x1, x0])
                    mesh_y.extend([y0, y1, y1, y0])
                    mesh_z.extend([0, 0, bldg_height, bldg_height])
                    
                    # 2 triangles to form the rectangular wall
                    mesh_i.extend([vertex_offset, vertex_offset])
                    mesh_j.extend([vertex_offset+1, vertex_offset+2])
                    mesh_k.extend([vertex_offset+2, vertex_offset+3])
                    
                    vertex_offset += 4
                
                # Triangulate the flat roof
                try:
                    tris = triangulate(poly)
                    for t in tris:
                        # Only include triangles that are strictly inside the polygon (handles concavity)
                        if poly.contains(t.centroid):
                            tx, ty = list(t.exterior.xy[0])[:3], list(t.exterior.xy[1])[:3]
                            mesh_x.extend(tx)
                            mesh_y.extend(ty)
                            mesh_z.extend([bldg_height] * 3)
                            
                            mesh_i.append(vertex_offset)
                            mesh_j.append(vertex_offset+1)
                            mesh_k.append(vertex_offset+2)
                            
                            vertex_offset += 3
                except Exception:
                    pass

        fig.add_trace(go.Mesh3d(
            x=mesh_x, y=mesh_y, z=mesh_z,
            i=mesh_i, j=mesh_j, k=mesh_k,
            color='rgba(140, 140, 150, 1.0)',
            opacity=0.7,
            flatshading=True,
            name='3D Buildings',
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1, roughness=0.8)
        ))

    # 2. Plot BS-UE Connections (Dashed orange lines in 3D)
    link_x, link_y, link_z = [], [], []
    for sector_idx in range(sim.num_sectors):
        bs_idx = sector_parent[sector_idx]
        bs_pos = bs_positions[bs_idx]
        
        connected_devices = np.where(association_matrix[sector_idx] == 1)[0]
        for dev_idx in connected_devices:
            dev_pos = ue_positions[dev_idx]
            link_x.extend([bs_pos[0], dev_pos[0], None])
            link_y.extend([bs_pos[1], dev_pos[1], None])
            link_z.extend([bs_pos[2], dev_pos[2], None])

    if link_x:
        fig.add_trace(go.Scatter3d(
            x=link_x, y=link_y, z=link_z,
            mode='lines',
            line=dict(color="#ff0e0e", width=3),
            name='BS-UE Connection'
        ))

    # 3. Plot Base Stations (Red Diamonds in 3D)
    fig.add_trace(go.Scatter3d(
        x=bs_positions[:, 0],
        y=bs_positions[:, 1],
        z=bs_positions[:, 2],
        mode='markers',
        marker=dict(symbol='diamond', size=6, color='red', line=dict(color='black', width=1)),
        name='Base Stations',
        text=[f"Base Station {i} (Z={bs_positions[i, 2]:.1f}m)" for i in range(len(bs_positions))],
        hoverinfo='text'
    ))

    # 4. Plot UEs / Devices (Blue Circles in 3D)
    fig.add_trace(go.Scatter3d(
        x=ue_positions[:, 0],
        y=ue_positions[:, 1],
        z=ue_positions[:, 2],
        mode='markers',
        marker=dict(symbol='circle', size=4, color='#1f77b4', line=dict(color='white', width=0.5)),
        name='UEs (Devices)',
        text=[f"Device {i} (Z={ue_positions[i, 2]:.1f}m)" for i in range(len(ue_positions))],
        hoverinfo='text'
    ))

    # Layout Customization for 3D mapping
    fig.update_layout(
        title="Simulator Topology",
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
            aspectmode='data' 
        ),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1),
        template='plotly_white',
        width=1200,
        height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save to HTML file
    output_filename = "simulation_topology_3d.html"
    fig.write_html(output_filename)
    print(f"Visualization successfully saved to '{output_filename}'. Open this file in your web browser to view the 3D plot.")

if __name__ == "__main__":
    visualize_simulation_plotly_3d()