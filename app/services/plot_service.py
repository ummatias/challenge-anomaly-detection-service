"""
Plot Service

Provides functionality to generate HTML plots of training data for a given time series and model version.
It uses Plotly to create interactive visualizations that highlight normal data points andanomalies based
on a calculated threshold.

Alternatively, we could use Json to return the data and let the frontend handle the plotting or Matplotlib/Seaborn
to generate static images, but Plotly offers a good balance of interactivity and ease of integration.
"""

import plotly.graph_objects as go

from app.core import persistence, versioning


class PlotService:

    def plot(self, series_id: str, version: str | None = None) -> str:
        manifest = persistence.load_manifest(series_id)
        if not manifest.get("versions"):
            raise LookupError(f"No model found for '{series_id}'.")

        resolved = version or versioning.latest_version(manifest)
        if not versioning.version_exists(manifest, resolved):
            raise LookupError(f"Version '{resolved}' not found for '{series_id}'.")

        try:
            raw = persistence.load_training_data(series_id, resolved)
        except FileNotFoundError as exc:
            raise LookupError(str(exc)) from exc

        entry = versioning.get_version_entry(manifest, resolved)
        timestamps = raw["timestamps"]
        values = raw["values"]
        mean = entry["mean"]
        std = entry["std"]
        threshold = mean + 3 * std

        is_anomaly = [v > threshold for v in values]
        normal = [(t, v) for t, v, a in zip(timestamps, values, is_anomaly) if not a]
        anomalies = [(t, v) for t, v, a in zip(timestamps, values, is_anomaly) if a]

        fig = go.Figure()
        fig.add_scatter(
            x=[t for t, _ in normal],
            y=[v for _, v in normal],
            mode="markers",
            name="Normal",
            marker=dict(color="#60a5fa", size=6),
        )
        fig.add_scatter(
            x=[t for t, _ in anomalies],
            y=[v for _, v in anomalies],
            mode="markers",
            name="Anomaly",
            marker=dict(
                color="#f87171",
                size=9,
                symbol="circle-open",
                line=dict(width=2, color="#f87171"),
            ),
        )
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#f87171",
            line_width=1.5,
            annotation_text=f"threshold ({threshold:.3f})",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"{series_id} · {resolved}  —  {len(values)} points, {sum(is_anomaly)} anomalies",
            xaxis_title="Timestamp",
            yaxis_title="Value",
            template="plotly_dark",
            margin=dict(t=60, r=20, b=50, l=60),
        )
        return fig.to_html(full_html=True, include_plotlyjs="cdn")


plot_service = PlotService()


def get_plot_service() -> PlotService:
    return plot_service
