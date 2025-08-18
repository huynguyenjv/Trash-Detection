# Smart Waste Detection System - Frontend

React-based frontend for the Smart Waste Detection System with real-time detection, waste statistics, and interactive mapping.

## Features

- **Real-time Video Stream**: Live camera feed with YOLOv8 detection overlay
- **WebSocket Integration**: Real-time detection data from backend
- **Waste Statistics**: Live dashboard showing waste categories and counts
- **Interactive Map**: Leaflet.js map with waste locations, bins, and optimal routing
- **Responsive Design**: Modern UI built with TailwindCSS

## Tech Stack

- **React 19** - Frontend framework
- **Vite** - Build tool and dev server
- **TailwindCSS** - Styling framework
- **Leaflet.js** - Interactive maps
- **WebSocket** - Real-time communication

## Components

### VideoStream.jsx
- Camera access and video display
- WebSocket connection for real-time detection
- Detection overlay with bounding boxes and labels
- Connection status indicators

### WasteStats.jsx
- Live statistics dashboard
- Waste categorization (organic, recyclable, hazardous, other)
- Progress bars and visual indicators
- Auto-refresh functionality

### MapView.jsx
- Interactive map with OpenStreetMap tiles
- Waste location markers
- Bin location markers
- Route calculation and display
- GPS integration

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

## Build

```bash
npm run build
```

## Backend Integration

The frontend expects a FastAPI backend running on `http://localhost:8000` with the following endpoints:

- `WebSocket /ws/detect` - Real-time detection data
- `GET /stats` - Waste statistics
- `GET /path` - Route calculation

## Configuration

The app connects to backend endpoints that can be configured in the component files:
- WebSocket endpoint in `VideoStream.jsx`
- API endpoints in `WasteStats.jsx` and `MapView.jsx`+ Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
