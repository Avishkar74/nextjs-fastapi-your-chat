/** @type {import('next').NextConfig} */
const nextConfig = {
  rewrites: async () => {
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/api/:path*"
            : `${process.env.NEXT_PUBLIC_API_URL || 'https://your-backend-url.railway.app'}/api/:path*`,
      },
      {
        source: "/docs",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/docs"
            : `${process.env.NEXT_PUBLIC_API_URL || 'https://your-backend-url.railway.app'}/docs`,
      },
      {
        source: "/openapi.json",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/openapi.json"
            : `${process.env.NEXT_PUBLIC_API_URL || 'https://your-backend-url.railway.app'}/openapi.json`,
      },
    ];
  },
};

module.exports = nextConfig;
