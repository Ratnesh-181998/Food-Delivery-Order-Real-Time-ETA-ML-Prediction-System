# Zomato ETA Prediction - React Frontend

This is the interactive React.js frontend for the Zomato ETA Prediction ML System Design project.

## 🚀 Features

- **Live Demo**: Interactive ETA prediction with real-time results
- **System Architecture**: Visual representation of AWS-based ML pipeline
- **Feature Engineering**: Detailed breakdown of all features used
- **Model Performance**: Comprehensive metrics and model comparison

## 📦 Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

## 🏗️ Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` directory.

## 🎨 Tech Stack

- **React 18**: Modern React with hooks
- **CSS3**: Custom styling with CSS variables
- **Recharts**: Data visualization
- **Leaflet**: Interactive maps
- **Lucide React**: Modern icons

## 📁 Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Header.js
│   │   ├── LiveDemo.js
│   │   ├── ETAPredictionForm.js
│   │   ├── PredictionResult.js
│   │   ├── SystemArchitecture.js
│   │   ├── FeatureExplorer.js
│   │   ├── ModelPerformance.js
│   │   └── Footer.js
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
└── package.json
```

## 🎯 Components

### Header
- Zomato branding
- Navigation links
- Key metrics banner

### LiveDemo
- Interactive prediction form
- Real-time results display
- How-it-works section

### SystemArchitecture
- AWS services visualization
- Architecture layers
- Data flow diagram

### FeatureExplorer
- Feature categories breakdown
- Formulas and examples
- Engineering pipeline

### ModelPerformance
- Performance metrics
- Model comparison table
- Feature importance chart

## 🎨 Design System

The app uses a comprehensive design system with:
- Zomato brand colors (#E23744)
- Modern glassmorphism effects
- Smooth animations and transitions
- Fully responsive layout

## 📱 Responsive Design

The UI is fully responsive and works on:
- Desktop (1400px+)
- Tablet (768px - 1399px)
- Mobile (< 768px)

## 🔧 Customization

### Colors
Edit CSS variables in `src/index.css`:
```css
:root {
  --zomato-red: #E23744;
  --primary: #E23744;
  /* ... more variables */
}
```

### API Integration
To connect to a real backend API, update the `handlePrediction` function in `App.js`:
```javascript
const response = await fetch('YOUR_API_ENDPOINT', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(formData)
});
```

## 📄 License

MIT License - see LICENSE file for details

## 👥 Author

ML System Design Project - Zomato ETA Prediction
