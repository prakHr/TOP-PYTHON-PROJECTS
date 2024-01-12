import geopandas
import libpysal
from libpysal.cg import Point, Chain
import matplotlib
import matplotlib_scalebar
from matplotlib_scalebar.scalebar import ScaleBar
import spaghetti

def operate_spatial_network(lines,fns):
	ntw = spaghetti.Network(in_data = lines)
	rv = {}
	for fn in fns:rv[fn] = eval(f"ntw.{fn}")

	vertices_df,arcs_df = spaghetti.element_as_gdf(ntw,vertices=True,arcs = True)
	rv["vertices_df"] = vertices_df
	rv["arcs_df"] = arcs_df
	napts = -1
	if "non_articulation_points" in fns:
		napts = rv["non_articulation_points"]
	else:
		napts = ntw.non_articulation_points

	articulation_vertices = vertices_df[~vertices_df["id"].isin(napts)]
	non_articulation_vertices = vertices_df[vertices_df["id"].isin(napts)]
	rv["articulation_vertices"] = articulation_vertices
	rv["non_articulation_vertices"] = non_articulation_vertices

	# Filter non-longest component arcs
	nlc = ntw.network_longest_component
	arcs_df = arcs_df[arcs_df.comp_label != nlc]
	ocomp = list(set(ntw.network_component_labels))
	ocomp.remove(nlc)
	rv["non_longest_arcs_component"] = ocomp
	return rv
	

if __name__=="__main__":
	lines = [
	    Chain([Point([1, 2]), Point([0, 2])]),
	    Chain([Point([1, 2]), Point([1, 1])]),
	    Chain([Point([1, 2]), Point([1, 3])]),
	]
	fns = [
		"non_articulation_points",
		"network_fully_connected",
		"network_n_components",
		"network_component2arc",
		"network_component_labels",
		"network_component2arc",
		"network_component_lengths",
		"network_longest_component",
		"network_component_vertices",
		"network_component_vertex_count",
		"network_largest_component",
		"network_component_is_ring",
		"graph_component_labels",
		"graph_component2edge"

	]
	rv = operate_spatial_network(lines,fns)
	from pprint import pprint
	pprint(rv)