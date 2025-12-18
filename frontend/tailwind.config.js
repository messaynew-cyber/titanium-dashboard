/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // The Deep Black/Green Background
        background: "#000d01",
        
        // Glassy Cards (Slightly lighter dark green)
        surface: "#051306", 
        
        // Text Colors
        primary: "#FFFFFF",
        secondary: "#9ca3af", // Neutral gray for labels
        
        // The Neon Green Accent (From your CSS variable --base-color-brand--green)
        accent: "#53db78",
        
        // Semantic Colors
        success: "#53db78", // Matching the accent
        danger: "#ff5050",  // From your CSS error variables
        warning: "#F59E0B",
        
        // Dark Borders
        border: "#1f2923",
      },
      fontFamily: {
        sans: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
