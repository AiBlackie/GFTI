# GFTI - Global Financial Transparency Index

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive dashboard that measures and visualizes sovereign financial reporting quality across countries, quantifying the **"Trust Tax"** - the additional borrowing costs countries pay due to transparency concerns.

![GFTI Dashboard Preview](https://via.placeholder.com/800x400?text=GFTI+Dashboard+Preview)

## 📊 Overview

The Global Financial Transparency Index (GFTI) provides a systematic framework for assessing the quality and reliability of sovereign financial reporting. By analyzing audit opinions, financial disclosures, and market data, the platform calculates:

- **GFTI Scores (0-100)** - Quantitative measure of financial transparency
- **Letter Grades (AAA to D)** - Easy-to-understand rating system  
- **Trust Tax** - The premium investors charge due to opacity (in basis points)
- **Annual Cost Impact** - Dollar cost of transparency gaps on sovereign debt

## ✨ Features

### 📈 Interactive Dashboard
- Real-time visualization of GFTI scores vs bond yields
- Color-coded by S&P ratings with bubble sizes representing debt/GDP
- Trend line analysis showing the ~10bps per GFTI point relationship

### 🔍 Country Deep Dive
- Comprehensive country profiles with key metrics
- Automated report generation in professional format
- Audit opinion analysis and key issue identification

### 🧮 Trust Tax Calculator
- Calculate the cost of opacity for any country
- Input S&P rating and GFTI score
- See expected yield vs actual yield
- Annual cost impact on $1B debt

### 📥 Data Management
- Upload new countries via CSV template
- Download full datasets
- Export individual country reports
- Batch report generation capability

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/gfti-dashboard.git
cd gfti-dashboard
Install dependencies

bash
pip install -r requirements.txt
Run the app

bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501

📁 Project Structure
text
gfti-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── countries.csv          # Country data (optional - app includes fallback data)
├── README.md              # This file
└── .gitignore            # Git ignore file
📋 Data Format
The application expects a CSV file with the following columns:

Column	Description	Example
Country	Country name	"Barbados"
Region	Geographic region	"Caribbean"
GFTI_Score	Transparency score (0-100)	18
SP_Rating	S&P credit rating	"B+"
SP_Numeric	Numeric rating (1-22)	14
Bond_Yield	10-year bond yield (%)	8.0
Yield_Type	Yield data source	"Govt 10Y"
Debt_GDP	Debt to GDP ratio (%)	102.9
GDP_Growth	GDP growth rate (%)	4.2
Inflation	Inflation rate (%)	4.5
Audit_Opinion	Audit opinion type	"Adverse"
Key_Issue	Main transparency issue	"$2.43B unverified receivables"
SOE_Consolidated	SOE consolidation status	"No"
Pension_Recorded	Pension liability recorded	"No"
Report_File	Associated report file	(optional)
Note: The app includes fallback data for 10 countries if no CSV is provided.

🎯 Grade Scale
Score Range	Grade	Risk Level
90-100	AAA	Minimal risk
80-89	AA	Low risk
70-79	A	Low-moderate risk
60-69	BBB	Moderate risk
50-59	BB	Moderate-high risk
40-49	B	High risk
30-39	CCC	Very high risk
20-29	CC	Extremely high risk
10-19	C	Critical risk
0-9	D	Default risk
🧮 Trust Tax Calculation
The Trust Tax model uses a simple formula:

Base expectation: 2.5% + (SP_Numeric × 0.35%)

GFTI adjustment: -10 basis points per GFTI point

Trust Tax: Actual Yield - Expected Yield (with GFTI adjustment)

🚀 Deployment
Deploy to Streamlit Cloud
Push your code to GitHub

Go to share.streamlit.io

Connect your GitHub repository

Deploy with default settings

Deploy to Heroku
Add a Procfile:

text
web: streamlit run app.py --server.port $PORT
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Created by: Matthew A A Blackman

With technical assistance from DeepSeek - an AI assistant helping to develop and refine the application code, documentation, and implementation.

🙏 Acknowledgments
Auditor General reports from various countries

S&P sovereign rating methodologies

World Bank Open Data

IMF World Economic Outlook

📧 Contact
For questions or feedback about the GFTI Dashboard:

Matthew A A Blackman

GitHub: aiblackie

Email: matthew.aa.blackman@gmail.com

Disclaimer: This tool is for informational purposes only. The Trust Tax calculation uses a preliminary model and should not be used as the sole basis for investment decisions
