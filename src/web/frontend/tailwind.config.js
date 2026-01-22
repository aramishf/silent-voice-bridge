/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'asl-primary': '#4F46E5',  // Indigo 600
                'asl-dark': '#111827',     // Gray 900
                'asl-card': '#1F2937',     // Gray 800
            },
            fontFamily: {
                'sans': ['Inter', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
