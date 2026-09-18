/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          page: "#0d0d0d",
          card: "#1a1a19",
          raised: "#232322",
        },
        ink: {
          primary: "#ffffff",
          secondary: "#c3c2b7",
          muted: "#898781",
        },
        line: {
          hairline: "#2c2c2a",
          baseline: "#383835",
          border: "rgba(255,255,255,0.10)",
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
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
