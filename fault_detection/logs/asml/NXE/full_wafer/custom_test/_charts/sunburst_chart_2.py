import pandas as pd
import plotly.express as px
import os

class SunburstChartPX:
    def __init__(self, excel_path, sheet_name=0):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = pd.read_excel(excel_path, sheet_name=sheet_name)
        self.root = "NXE"
        self._clean_features()

    def _clean_features(self):
        def clean_feature_string(s):
            s = str(s)
            s = s.replace('[', '').replace(']', '')
            s = s.replace('feat_list=', '')
            s = s.replace('_', ' ')
            s = s.replace('(', ' (')
            s = s.strip()
            s = s.capitalize()
            return s
        self.df['Features'] = self.df['Features'].apply(clean_feature_string)

    def plot(self, save_path=None):
        # Build hierarchy path
        # If you want to include the root, add a column
        self.df['Root'] = self.root

        # You can choose which columns to use for the hierarchy
        path = ['Root', 'Module', 'Group', 'Features', 'FP (H)', 'FP (S)']

        # Choose color column (e.g., Module for categorical, FP (H) for continuous)
        fig = px.sunburst(
            self.df,
            path=path,
            values='FP (H)',  # or another column, or None
            color='Module',   # color by module for legend
            color_discrete_sequence=px.colors.qualitative.Set2,  # choose your palette
            # color_continuous_scale=px.colors.sequential.Blues, # for continuous
        )

        # Set font and layout for IEEE style
        fig.update_layout(
            font=dict(
                family="Times New Roman, Times, serif",
                size=16,
                color="black"
            ),
            width=1000,
            height=1000,
        )

        # Save high quality PNG if requested
        if save_path:
            fig.write_image(save_path, format="png", width=2000, height=2000, scale=4)

        import plotly.io as pio
        pio.renderers.default = 'browser'

        fig.show()

if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), 'fdet_asml.xlsx')
    chart = SunburstChartPX(excel_path)
    chart.plot(save_path=os.path.join(os.path.dirname(excel_path), "sunburst_chart_px.png"))