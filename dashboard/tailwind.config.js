/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          page: "#fdf9eb",
          card: "#ffffff",
          raised: "#f3eedd",
        },
        ink: {
          primary: "#232322",
          secondary: "#605f5a",
          muted: "#908e87",
        },
        line: {
          hairline: "#e3dfd3",
          baseline: "#d1cec3",
          border: "rgba(35,35,34,0.14)",
        },
        status: {
          good: "#0ca30c",
          warning: "#fab219",
          serious: "#ec835a",
          critical: "#e66767",
        },
        series: {
          blue: "#3987e5",
          orange: "#d95926",
          aqua: "#199e70",
          yellow: "#c98500",
          magenta: "#d55181",
          green: "#008300",
          violet: "#9085e9",
          red: "#e66767",
        },
        // LTA brand navy — header/nav band only, not the content palette.
        brand: {
          navy: "#171c8f",
        },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
