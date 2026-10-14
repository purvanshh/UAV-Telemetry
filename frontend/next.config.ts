import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Dev uses Turbopack (`next dev --turbopack`); keep alias there, not only webpack.
  turbopack: {
    resolveAlias: {
      "plotly.js/dist/plotly": "plotly.js/dist/plotly.min.js",
    },
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      "plotly.js/dist/plotly": "plotly.js/dist/plotly.min.js",
    };
    return config;
  },
};

export default nextConfig;
