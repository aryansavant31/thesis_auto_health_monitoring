import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

fig = go.Figure(go.Sunburst(
    labels=["NXE", "Module1", "Group1"],
    parents=["", "NXE", "Module1"],
    values=[0, 0, 10]
))
fig.show()