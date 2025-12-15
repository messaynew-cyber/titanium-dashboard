/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#F8F9FA", surface: "#FFFFFF", primary: "#0F172A",
        secondary: "#64748B", accent: "#3B82F6", success: "#10B981",
        danger: "#EF4444", warning: "#F59E0B", border: "#E2E8F0",
      },
    },
  },
  plugins: [],
}