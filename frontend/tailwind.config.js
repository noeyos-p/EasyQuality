/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#1F1F1F',
          deeper: '#181818',
          light: '#252526',
          hover: '#2a2d2e',
          border: '#3e3e42',
          active: '#37373d',
        },
        txt: {
          primary: '#cccccc',
          secondary: '#858585',
          muted: '#6a6a6a',
          white: '#ffffff',
        },
        accent: {
          DEFAULT: '#22D142',
          hover: '#1ab835',
          blue: '#007acc',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideInFromRight: {
          '0%': { transform: 'translateX(1rem)' },
          '100%': { transform: 'translateX(0)' },
        },
      },
      animation: {
        'fade-in': 'fadeIn 300ms ease-out forwards',
        'slide-in-from-right': 'slideInFromRight 300ms ease-out forwards',
      },
    },
  },
  plugins: [],
}
