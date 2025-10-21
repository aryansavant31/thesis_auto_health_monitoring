import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class SunburstChart:
    def __init__(self, excel_path, sheet_name=0):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = pd.read_excel(excel_path, sheet_name=0)
        self.root = 'NXE'
        self.ids = []
        self.display_labels = []
        self.parents = []
        self.values = []
        self.colors = []
        self._prepare_data()

    @staticmethod
    def clean_feature_string(s):
        s = str(s)
        s = s.replace('[', '').replace(']', '')
        s = s.replace('feat_list=', '')
        s = s.replace('_', ' ')
        s = s.replace('(', ' (')
        s = s.strip()
        s = s.capitalize()
        return s

    @staticmethod
    def value_to_alpha_color(val, min_val, max_val, base_color, alpha_min=0.0, alpha_max=0.7):
        if max_val == min_val:
            alpha = alpha_max
        else:
            alpha = alpha_min + (alpha_max - alpha_min) * ((val - min_val) / (max_val - min_val))
        rgba = (base_color[0], base_color[1], base_color[2], alpha)
        return f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"

    def hex_to_rgb_str(self, hex_color):
        rgb = mcolors.to_rgb(hex_color)
        return f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"

    def _prepare_data(self):
        df = self.df
        df['Features'] = df['Features'].apply(self.clean_feature_string)

        modules = df['Module'].unique()
        palette = plt.get_cmap('tab10_r')
        module_colors = [palette(i) for i in range(len(modules))]  # RGBA tuples

        group_shades = {}
        module_color_map = {}
        for i, module in enumerate(modules):
            groups = df[df['Module'] == module]['Group'].unique()
            base_color = module_colors[i]  # RGBA tuple
            module_color_map[module] = f"rgba({int(base_color[0]*255)},{int(base_color[1]*255)},{int(base_color[2]*255)},1.0)"
            n = len(groups)
            if n == 1:
                alpha = 1
                rgba = (base_color[0], base_color[1], base_color[2], alpha)
                rgb_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"
                group_shades[module] = [rgb_str]
            elif n > 1:
                alphas = [0.3 + 0.2 * (j / (n-1)) for j in range(n)]  # from 0.6 to 0.9
                group_colors = []
                for alpha in alphas:
                    rgba = (base_color[0], base_color[1], base_color[2], alpha)
                    rgb_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"
                    group_colors.append(rgb_str)
                group_shades[module] = group_colors

        fp_h_min, fp_h_max = 0, 10
        fn_h_min, fn_h_max =  0, 10
        fp_s_min, fp_s_max =  0, 10
        fn_s_min, fn_s_max =  0, 10
        # fp_h_min, fp_h_max = df['FP (H)'].min(), df['FP (H)'].max()
        # fn_h_min, fn_h_max = df['FN (H)'].min(), df['FN (H)'].max()
        # fp_s_min, fp_s_max = df['FP (S)'].min(), df['FP (S)'].max()
        # fn_s_min, fn_s_max = df['FN (S)'].min(), df['FN (S)'].max()

        ids, display_labels, parents, values, colors = [], [], [], [], []

        module_group_pairs = set()
        group_feature_pairs = set()

        # Add root node
        ids.append("root")
        display_labels.append(self.root)
        parents.append("")
        values.append(0)
        colors.append("white")

        for i, row in df.iterrows():
            module = row['Module']
            group = row['Group']
            features = row['Features']
            fp_h = row['FP (H)']
            fn_h = row['FN (H)']
            fp_s = row['FP (S)']
            fn_s = row['FN (S)']

            if features != "Raw time data":
                continue

            # Level 2: Module
            module_id = f"module_{module}"
            blank_module_id = f"blank_{module_id}"
            if module_id not in ids:
                ids.append(module_id)
                display_labels.append(module)
                parents.append("root")
                values.append(0)
                colors.append(module_color_map[module])

                # Add blank node under module
                ids.append(blank_module_id)
                display_labels.append(f"Module")  # Blank label
                parents.append(module_id)
                values.append(0)
                colors.append("white")

            # For blue base color:
            base_color_red = (1.0, 0.0, 0.0, 1.0)  # RGBA tuple for red

            # For yellow base color:
            base_color_yellow = (1.0, 0.85, 0.0, 1.0)

            # Level 3: Group (unique per module)
            group_id = f"group_{group}_{module}"
            blank_group_id = f"blank_{group_id}"
            if group_id not in ids:
                ids.append(group_id)
                display_labels.append(group)
                parents.append(blank_module_id)  # <-- group is child of blank_module_id
                values.append(0)
                colors.append(group_shades[module][list(df[df['Module'] == module]['Group'].unique()).index(group)])

                # Add blank node under group
                ids.append(blank_group_id)
                display_labels.append("")  # Blank label
                parents.append(group_id)
                values.append(0)
                colors.append("white")

            # Level 5: FP (H) / FN (H)
            fp_h_id = f"fp_h_{i}_{group_id}"
            fn_h_id = f"fn_h_{i}_{group_id}"
            ids.extend([fp_h_id, fn_h_id])
            display_labels.extend([f"FP (H): {fp_h}", f"FN (H): {fn_h}"])
            parents.extend([blank_group_id, blank_group_id])  # <-- FP/FN are children of blank_group_id
            values.extend([fp_h, fn_h])
            colors.extend([
                self.value_to_alpha_color(fp_h, fp_h_min, fp_h_max, base_color_red, alpha_min=0.0, alpha_max=0.7),
                self.value_to_alpha_color(fn_h, fn_h_min, fn_h_max, base_color_red, alpha_min=0.0, alpha_max=0.7)
            ])

            # Level 6: FP (S) / FN (S)
            fp_s_id = f"fp_s_{i}_{fp_h_id}"
            fn_s_id = f"fn_s_{i}_{fn_h_id}"
            ids.extend([fp_s_id, fn_s_id])
            display_labels.extend([f"FP (S): {fp_s}", f"FN (S): {fn_s}"])
            parents.extend([fp_h_id, fn_h_id])
            values.extend([fp_s, fn_s])
            colors.extend([
                self.value_to_alpha_color(fp_s, fp_s_min, fp_s_max, base_color_yellow, alpha_min=0.0, alpha_max=0.7),
                self.value_to_alpha_color(fn_s, fn_s_min, fn_s_max, base_color_yellow, alpha_min=0.0, alpha_max=0.7)
            ])

        self.ids = ids
        self.display_labels = display_labels
        self.parents = parents
        self.values = values
        self.colors = colors

    def plot(self):
        import plotly.io as pio
        pio.renderers.default = 'browser'

        fig = go.Figure(go.Sunburst(
            ids=self.ids,
            labels=self.display_labels,
            parents=self.parents,
            #values=self.values,
            marker=dict(
                colors=self.colors,
                line=dict(color='grey', width=2)  # <-- Add this line
            ),
            hovertext=self.display_labels,
            branchvalues='total'
        ))

        fig.update_layout(
            font=dict(
                family="Times New Roman, Times, serif",
                size=16,
                color="black"
            ),
        )
        fig.write_image(os.path.join(os.path.dirname(self.excel_path), "sunburst_chart_raw_time.png"), format="png", width=1000, height=1000, scale=6)  # High quality PNG
        fig.show()

if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), 'fdet_asml.xlsx')
    chart = SunburstChart(excel_path)
    chart.plot()