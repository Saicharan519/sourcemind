/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Editorial / Library — warm paper + ink, emerald + coral accents
        paper: "#F4EEE1",
        "paper-raised": "#FBF7EE",
        "paper-sunken": "#EDE5D3",
        ink: "#1B2233",
        "ink-soft": "#5C5A55",
        "ink-faint": "#8A857A",
        line: "#E2D8C4",
        "line-strong": "#D4C7AC",
        emerald: "#0F7A55",
        "emerald-deep": "#0A5C40",
        "emerald-soft": "#D8E8DF",
        coral: "#E1573A",
        "coral-soft": "#F6DDD3",
        gold: "#C08A2D",
      },
      fontFamily: {
        display: ['"Fraunces"', "Georgia", "serif"],
        sans: ['"Instrument Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(27,34,51,0.04), 0 8px 24px -12px rgba(27,34,51,0.16)",
        "card-hover":
          "0 2px 4px rgba(27,34,51,0.06), 0 18px 40px -16px rgba(27,34,51,0.28)",
        inset: "inset 0 1px 0 rgba(255,255,255,0.6)",
      },
      borderRadius: {
        xl: "0.9rem",
        "2xl": "1.25rem",
      },
      keyframes: {
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        drift: {
          "0%, 100%": { transform: "translate(0, 0) scale(1)" },
          "50%": { transform: "translate(3%, -4%) scale(1.08)" },
        },
      },
      animation: {
        blink: "blink 1.1s step-start infinite",
        drift: "drift 18s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
