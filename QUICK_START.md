# 🚀 Quick Start Guide - Zomato ETA Prediction System

## Prerequisites

- **Node.js** 16+ and npm
- **Python** 3.8+
- **Git**

## 🎯 Running the React Frontend

### Step 1: Navigate to Frontend Directory
```powershell
cd C:\Users\rattu\Downloads\L-9\frontend
```

### Step 2: Install Dependencies
```powershell
npm install
```

### Step 3: Start Development Server
```powershell
npm start
```

The application will automatically open in your browser at **http://localhost:3001**

## 🐍 Running the Python Backend (Optional)

### Step 1: Create Virtual Environment
```powershell
cd C:\Users\rattu\Downloads\L-9
python -m venv venv
.\venv\Scripts\activate
```

### Step 2: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Feature Engineering Demo
```powershell
python src\data\feature_engineering.py
```

### Step 4: Run Model Training Demo
```powershell
python src\models\model_trainer.py
```

## 📊 What You'll See

### 1. Live Demo Tab
- Interactive form to input order details
- Real-time ETA predictions
- Breakdown of prediction components
- Feature visualization

### 2. System Architecture Tab
- AWS services used
- Architecture layers
- Data flow diagram
- Key features

### 3. Feature Engineering Tab
- Distance features (Haversine formula)
- Temporal features (time-based)
- Restaurant & rider features
- Traffic & weather features
- Feature engineering pipeline

### 4. Model Performance Tab
- Performance metrics (MAE, RMSE, R²)
- Model comparison table
- Feature importance chart
- Deployment configuration

## 🎨 Features

✅ **Modern UI** - Beautiful, responsive design with Zomato branding  
✅ **Interactive Demo** - Test ETA predictions with custom inputs  
✅ **Comprehensive Documentation** - Full system design and architecture  
✅ **Real-time Visualization** - Charts, graphs, and metrics  
✅ **Mobile Responsive** - Works on all devices  

## 🛠️ Troubleshooting

### Port Already in Use
If port 3000 is busy:
```powershell
# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use a different port
set PORT=3001 && npm start
```

### Module Not Found
```powershell
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📚 Next Steps

1. ✅ Explore the Live Demo
2. ✅ Review System Architecture
3. ✅ Study Feature Engineering
4. ✅ Analyze Model Performance
5. ✅ Read documentation in `/docs`

## 🔗 Important Files

- `frontend/` - React.js UI
- `src/data/feature_engineering.py` - Feature engineering code
- `src/models/model_trainer.py` - Model training code
- `src/api/lambda_handler.py` - AWS Lambda function
- `docs/SYSTEM_DESIGN.md` - Complete system design
- `docs/HYPOTHESIS_TESTING.md` - A/B testing guide
- `docs/CLASS_NOTES_REFERENCE.md` - Class notes summary

## 💡 Tips

- Use preset locations in the demo for quick testing
- Check the browser console for any errors
- All predictions are simulated (no real API calls)
- Explore all 4 tabs for complete understanding

## 🎓 Learning Resources

- AWS SageMaker: https://aws.amazon.com/sagemaker/
- XGBoost: https://xgboost.readthedocs.io/
- React.js: https://react.dev/
- Zomato Blog: https://blog.zomato.com/

---

**Ready to start?** Run `npm start` in the frontend directory! 🚀
