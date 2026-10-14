"use client";

import dynamic from "next/dynamic";
import type { Data, Layout, Config } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="chart-loading">Loading chart…</div>,
});

type PlotlyFigure = {
  data?: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
};

export function PlotChart({ figure }: { figure: PlotlyFigure | null }) {
  if (!figure) {
    return <div className="chart-loading">No chart data</div>;
  }
  return (
    <div className="chart-wrap">
      <Plot
        data={figure.data || []}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(12,17,23,0.4)",
          font: { color: "#c8d4e4", family: "DM Sans, sans-serif" },
          margin: { l: 48, r: 24, t: 48, b: 40 },
          ...(figure.layout || {}),
          autosize: true,
        }}
        config={{
          ...(figure.config || {}),
          displayModeBar: false,
          displaylogo: false,
          responsive: true,
          showLink: false,
          // Keep Plotly from probing this app / API host for /json/version
          plotlyServerURL: "https://chart-studio.plotly.com",
        }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
