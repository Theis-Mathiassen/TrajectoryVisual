import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, nearest_points
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def plot_trajectory_closest(origin_coords, similar_coords, dissimilar_coords, aura_delta_val, output_filename="similarity_distribution_closest.pdf"):
    """
    Plots trajectories for the "Closest" similarity method.
    All similar nodes are marked '1'. Dotted lines connect each origin node
    to its closest similar node. Delta arrow from origin_coords[1] to the window edge.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    color_origin_line = '#FF7F50'  
    color_origin_node = '#F08080'  
    color_origin_node_edge = '#CD5C5C' 

    color_similar_line = '#98FB98' 
    color_similar_node = '#90EE90' 
    color_similar_node_edge = '#3CB371' 
    
    color_dissimilar_line = '#B0E0E6' 
    color_dissimilar_node = '#87CEFA' 
    color_dissimilar_node_edge = '#4682B4' 
    
    color_aura_fill = '#FFF0F5'  
    color_aura_edge = '#FFB6C1'  

    color_node_text_origin = 'white' 
    color_node_text_similar = 'black'
    color_node_text_dissimilar = 'black'
    
    color_delta_arrow = 'black'
    color_dotted_lines = 'gray'


    origin_line = LineString(origin_coords)
    similar_line = LineString(similar_coords)
    dissimilar_line = LineString(dissimilar_coords)

    aura_polygon = origin_line.buffer(aura_delta_val, cap_style=1, join_style=1)
    if aura_polygon.geom_type == 'MultiPolygon':
        aura_polygon = unary_union(aura_polygon)

    # --- Plotting ---
    # Plot Aura
    aura_label_added = False
    if aura_polygon.geom_type == 'Polygon':
        aura_path = Path.make_compound_path(Path(np.asarray(aura_polygon.exterior.coords)[:, :2]))
        for interior in aura_polygon.interiors:
            aura_path.codes = np.concatenate([aura_path.codes, Path(np.asarray(interior.coords)[:, :2]).codes])
            aura_path.vertices = np.concatenate([aura_path.vertices, Path(np.asarray(interior.coords)[:, :2]).vertices])
        patch = PathPatch(aura_path, facecolor=color_aura_fill, alpha=0.7, edgecolor=color_aura_edge, linewidth=1.5, label='Window size / Similarity area', zorder=1)
        ax.add_patch(patch)
        aura_label_added = True
    elif aura_polygon.geom_type == 'MultiPolygon':
        for i, poly in enumerate(aura_polygon.geoms):
            aura_path = Path.make_compound_path(Path(np.asarray(poly.exterior.coords)[:, :2]))
            for interior in poly.interiors:
                 aura_path.codes = np.concatenate([aura_path.codes, Path(np.asarray(interior.coords)[:, :2]).codes])
                 aura_path.vertices = np.concatenate([aura_path.vertices, Path(np.asarray(interior.coords)[:, :2]).vertices])
            current_label = 'Window size / Similarity area' if i == 0 else None
            patch = PathPatch(aura_path, facecolor=color_aura_fill, alpha=0.7, edgecolor=color_aura_edge, linewidth=1.5, label=current_label, zorder=1)
            ax.add_patch(patch)
            if i == 0: aura_label_added = True
    if not aura_label_added: 
        ax.plot([], [], color=color_aura_fill, linewidth=10, label='Window size / Similarity area', alpha=0.7)

    # Plot Trajectories (lines)
    ax.plot(*origin_line.xy, color=color_origin_line, linewidth=3, solid_capstyle='round', zorder=3, label='_nolegend_')
    ax.plot(*similar_line.xy, color=color_similar_line, linewidth=3, solid_capstyle='round', zorder=3, label='_nolegend_')
    ax.plot(*dissimilar_line.xy, color=color_dissimilar_line, linewidth=3, solid_capstyle='round', zorder=2, label='_nolegend_')

    # Plot Nodes and Arrows for trajectories
    def plot_nodes_and_arrows(coords, node_fill_color, line_color, node_label_val=None, node_text_color='black', node_edge_color='black'):
        x_coords, y_coords = zip(*coords)
        for i in range(len(x_coords)):
            ax.plot(x_coords[i], y_coords[i], 'o', markersize=25, color=node_fill_color, markeredgecolor=node_edge_color, zorder=5, mew=1.5)
            if node_label_val is not None: 
                # Special case for second point in similar trajectory
                if node_label_val == 1 and i == 1 and coords == similar_coords:
                    label = "2"
                else:
                    label = str(node_label_val)
                ax.text(x_coords[i], y_coords[i], label, color=node_text_color, 
                        ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)
            if i < len(x_coords) - 1:
                ax.annotate("",
                            xy=(x_coords[i+1], y_coords[i+1]), xycoords='data',
                            xytext=(x_coords[i], y_coords[i]), textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=line_color, shrinkA=12, shrinkB=12, lw=2.5),
                            zorder=4)

    plot_nodes_and_arrows(origin_coords, color_origin_node, color_origin_line, node_text_color=color_node_text_origin, node_edge_color=color_origin_node_edge)
    plot_nodes_and_arrows(similar_coords, color_similar_node, color_similar_line, node_label_val=1, node_text_color=color_node_text_similar, node_edge_color=color_similar_node_edge)
    plot_nodes_and_arrows(dissimilar_coords, color_dissimilar_node, color_dissimilar_line, node_label_val=0, node_text_color=color_node_text_dissimilar, node_edge_color=color_dissimilar_node_edge)
    
    # --- Dotted Lines from Origin to Closest Similar Node ---
    similar_points_shapely = [Point(c) for c in similar_coords]
    for ox, oy in origin_coords:
        origin_p = Point(ox, oy)
        closest_geom_pair = nearest_points(origin_p, unary_union(similar_points_shapely))
        closest_similar_p = closest_geom_pair[1]
        if closest_similar_p:
             ax.plot([ox, closest_similar_p.x], [oy, closest_similar_p.y], 
                    linestyle=':', color=color_dotted_lines, lw=2, zorder=3.5)

    # --- Additional dotted line from second similar point to first origin point ---
    first_origin_point = Point(origin_coords[0])
    second_similar_point = Point(similar_coords[1])
    
    # Calculate distance between points
    distance = first_origin_point.distance(second_similar_point)
    
    # Only draw if within delta distance
    if distance <= aura_delta_val:
        ax.plot([first_origin_point.x, second_similar_point.x], 
                [first_origin_point.y, second_similar_point.y],
                linestyle=':', color=color_dotted_lines, lw=2, zorder=3.5)
        
        # Add delta symbol near the midpoint of the line
        mid_x = (first_origin_point.x + second_similar_point.x) / 2
        mid_y = (first_origin_point.y + second_similar_point.y) / 2
        ax.text(mid_x + 0.15, mid_y, 
                f'$\\leq\\Delta$', color=color_delta_arrow, fontsize=18, 
                ha='left', va='center', zorder=10, fontweight='bold')

    o2_point = Point(origin_coords[1]) 
    vertical_line = LineString([(o2_point.x, o2_point.y), (o2_point.x, o2_point.y + aura_delta_val * 2)])
    
    intersection_point_on_boundary = None
    if aura_polygon.exterior: 
        intersection = aura_polygon.exterior.intersection(vertical_line)
        if not intersection.is_empty:
            if intersection.geom_type == 'Point':
                intersection_point_on_boundary = (intersection.x, intersection.y)
            elif intersection.geom_type == 'MultiPoint':
                valid_points = [p for p in intersection.geoms if p.y > o2_point.y]
                if valid_points:
                    chosen_point = min(valid_points, key=lambda p: p.y)
                    intersection_point_on_boundary = (chosen_point.x, chosen_point.y)
            elif intersection.geom_type == 'LineString': 
                target_y = o2_point.y + aura_delta_val # Aim for the point aura_delta_val above o2
                min_dist_point = None
                min_y_diff = float('inf')
                # Find the point on the intersecting line segment that is closest to the target_y
                for coord_tuple in intersection.coords: 
                    if coord_tuple[1] >= o2_point.y and abs(coord_tuple[1] - target_y) < min_y_diff :
                        min_y_diff = abs(coord_tuple[1] - target_y)
                        min_dist_point = coord_tuple
                if min_dist_point:
                    intersection_point_on_boundary = min_dist_point
    
    # Fallback if intersection logic fails to find a good point (e.g. if aura is very complex)
    if not intersection_point_on_boundary:
        intersection_point_on_boundary = (o2_point.x, o2_point.y + aura_delta_val)


    if intersection_point_on_boundary:
        ax.annotate("",
                    xy=intersection_point_on_boundary, xycoords='data',
                    xytext=(o2_point.x, o2_point.y), textcoords='data',
                    arrowprops=dict(arrowstyle="<->", color=color_delta_arrow, shrinkA=0, shrinkB=0, lw=2.5), 
                    zorder=10)
        ax.text((o2_point.x + intersection_point_on_boundary[0]) / 2 + 0.25, 
                (o2_point.y + intersection_point_on_boundary[1]) / 2, 
                f'$\\Delta$', color=color_delta_arrow, fontsize=18, ha='left', va='center', zorder=10, fontweight='bold')

    # --- Styling and Legend ---
    ax.set_title('Similarity distribution: Closest + Furthest', fontsize=20, fontweight='bold', pad=25)
    ax.set_aspect('equal', adjustable='box')
    
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Origin trajectory',
                   markerfacecolor=color_origin_node, markeredgecolor=color_origin_node_edge, markersize=20, mew=1.5),
        plt.Line2D([0], [0], marker='o', color='w', label='Similar trajectory (node score: 1)', 
                   markerfacecolor=color_similar_node, markeredgecolor=color_similar_node_edge, markersize=20, mew=1.5),
        plt.Line2D([0], [0], marker='o', color='w', label='Dissimilar trajectory (node score: 0)',
                   markerfacecolor=color_dissimilar_node, markeredgecolor=color_dissimilar_node_edge, markersize=20, mew=1.5),
        plt.Rectangle((0,0),1,1, facecolor=color_aura_fill, edgecolor=color_aura_edge, alpha=0.7, linewidth=1.5, label='Window size / Similarity area')
    ]
    
    legend = ax.legend(handles=legend_handles, 
                       loc='upper right', 
                       fontsize=12, 
                       frameon=True, 
                       facecolor='white', 
                       framealpha=0.95,
                       shadow=True,
                       borderpad=0.8, 
                       labelspacing=0.8) 
    legend.get_frame().set_edgecolor('darkgray') 
    legend.get_frame().set_linewidth(1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {output_filename}")

    plt.tight_layout(pad=1.5) 
    # plt.show() 

# --- Define Coordinates ---
origin_trajectory = [(1.5, 2.5), (4, 3.2), (7, 2.2), (9.5, 3.5)]
similar_trajectory = [(0.7, 2), (2.1, 3.4), (6.2, 3.2), (10.5, 2.5)] 
dissimilar_trajectory = [(0.4, 1.6), (2.8, 0.6), (5.5, 1.4), (8.5, 1)]

# --- Define Aura Value ---
aura_buffer_distance = 1.5 

# --- Generate Plot ---
if __name__ == '__main__':
    plot_trajectory_closest(origin_coords=origin_trajectory, 
                            similar_coords=similar_trajectory, 
                            dissimilar_coords=dissimilar_trajectory, 
                            aura_delta_val=aura_buffer_distance,
                            output_filename='similarity_distribution_closestfurthest.pdf')
