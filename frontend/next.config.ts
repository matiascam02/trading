import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: false, // Disable to prevent DOM manipulation errors in dev
  typescript: {
    // Skip TypeScript errors during build
    ignoreBuildErrors: true
  },
  eslint: {
    // Skip ESLint during builds
    ignoreDuringBuilds: true
  },
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8000/api/v1/:path*',
      },
    ];
  },
};

export default nextConfig;
