import plotly.graph_objects as go

from pca_neural_trajectories import facet_figures


def test_facet_figures_keeps_hidden_traces_out_of_the_legend():
    panel = go.Figure(
        [
            go.Scatter(y=[0, 1], name="sub-01", showlegend=False),
            go.Scatter(y=[1, 2], name="Faces vs Scrambled"),
        ]
    )
    grid = facet_figures({"A": panel, "B": panel})
    shown = [trace.name for trace in grid.data if trace.showlegend]
    assert shown == ["Faces vs Scrambled"]
