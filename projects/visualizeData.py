# https://stackoverflow.com/questions/59630394/how-to-run-orange-canvas-from-within-a-python-script-and-transform-a-table-to-it
import Orange
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owbarplot import OWBarPlot
from Orange.widgets.visualize.owboxplot import OWBoxPlot
from Orange.widgets.visualize.owdistributions import OWDistributions
from Orange.widgets.visualize.owheatmap import OWHeatMap
from Orange.widgets.visualize.owlineplot import OWLinePlot
from Orange.widgets.visualize.owmosaic import OWMosaicDisplay
from Orange.widgets.visualize.owradviz import OWRadviz
from Orange.widgets.visualize.owscatterplot import OWScatterPlot
from Orange.widgets.visualize.owsilhouetteplot import OWSilhouettePlot
from Orange.widgets.visualize.owvenndiagram import OWVennDiagram
from Orange.widgets.visualize.owviolinplot import OWViolinPlot






def make_a_plot(dataset,plot_type = "OWBarPlot"):
    WidgetPreview(eval(plot_type)).run(set_data=dataset)




if __name__=="__main__":
    csv_path = r"C:\Users\gprak\Downloads\projects\Data\df_i.xlsx"
    dataset = Orange.data.Table(csv_path)
    # plot_types = [
        # "OWBarPlot",
        # "OWBoxPlot",
        # "OWViolinPlot",
        # "OWHeatMap",(set_dataset instead of set_data)
        # "OWLinePlot",
        # "OWScatterPlot",
        # "OWDistributions",
        # "OWMosaicDisplay",
        # "OWRadviz",
        # "OWSilhouettePlot",
    # ]
    # for plot_type in plot_types:
    plot_type = "OWBarPlot"
    make_a_plot(dataset,plot_type)