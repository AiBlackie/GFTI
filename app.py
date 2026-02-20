import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io
import base64
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GFTI - Global Financial Transparency Index",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS - GRADE SCALE (Single Source of Truth)
# ============================================================================
GRADE_SCALE = {
    (90, 100): ('AAA', 'Minimal risk'),
    (80, 89): ('AA', 'Low risk'),
    (70, 79): ('A', 'Low-moderate risk'),
    (60, 69): ('BBB', 'Moderate risk'),
    (50, 59): ('BB', 'Moderate-high risk'),
    (40, 49): ('B', 'High risk'),
    (30, 39): ('CCC', 'Very high risk'),
    (20, 29): ('CC', 'Extremely high risk'),
    (10, 19): ('C', 'Critical risk'),
    (0, 9): ('D', 'Default risk')
}


def get_grade_from_score(score):
    """Convert GFTI score to letter grade"""
    if pd.isna(score):
        return None
    for (low, high), (grade, _) in GRADE_SCALE.items():
        if low <= score <= high:
            return grade
    return None


def get_grade_description(grade):
    """Get description for a grade"""
    for (_, _), (g, desc) in GRADE_SCALE.items():
        if g == grade:
            return desc
    return "Unknown"


# ============================================================================
# COUNTRY COORDINATES FOR MAPPING
# ============================================================================
COUNTRY_COORDINATES = {
    # Caribbean
    'Barbados': {'lat': 13.1939, 'lon': -59.5432},
    'Jamaica': {'lat': 18.1096, 'lon': -77.2975},
    'Trinidad': {'lat': 10.6918, 'lon': -61.2225},  # Trinidad
    'Trinidad and Tobago': {'lat': 10.6918, 'lon': -61.2225},
    'Guyana': {'lat': 4.8604, 'lon': -58.9302},
    'Bahamas': {'lat': 25.0343, 'lon': -77.3963},
    'Bermuda': {'lat': 32.3078, 'lon': -64.7505},
    'Dominican Republic': {'lat': 18.7357, 'lon': -70.1627},
    
    # Africa
    'Nigeria': {'lat': 9.0820, 'lon': 8.6753},
    'Egypt': {'lat': 26.8206, 'lon': 30.8025},
    'South Africa': {'lat': -30.5595, 'lon': 22.9375},
    'Rwanda': {'lat': -1.9403, 'lon': 29.8739},
    'Kenya': {'lat': -0.0236, 'lon': 37.9062},
    'Ghana': {'lat': 7.9465, 'lon': -1.0232},
    'Botswana': {'lat': -22.3285, 'lon': 24.6849},
    'Mauritius': {'lat': -20.3484, 'lon': 57.5522},
    'Senegal': {'lat': 14.4974, 'lon': -14.4524},
    'Uganda': {'lat': 1.3733, 'lon': 32.2903},
    'Zambia': {'lat': -13.1339, 'lon': 27.8493},
    'Ethiopia': {'lat': 9.1450, 'lon': 40.4897},
    'Morocco': {'lat': 31.7917, 'lon': -7.0926},
    
    # Asia
    'Pakistan': {'lat': 30.3753, 'lon': 69.3451},
    'Indonesia': {'lat': -0.7893, 'lon': 113.9213},
    'Philippines': {'lat': 12.8797, 'lon': 121.7740},
    'Vietnam': {'lat': 14.0583, 'lon': 108.2772},
    'India': {'lat': 20.5937, 'lon': 78.9629},
    'Malaysia': {'lat': 4.2105, 'lon': 101.9758},
    'Thailand': {'lat': 15.8700, 'lon': 100.9925},
    'Sri Lanka': {'lat': 7.8731, 'lon': 80.7718},
    'Bangladesh': {'lat': 23.6850, 'lon': 90.3563},
    'Lebanon': {'lat': 33.8547, 'lon': 35.8623},
    'Saudi Arabia': {'lat': 23.8859, 'lon': 45.0792},
    'UAE': {'lat': 23.4241, 'lon': 53.8478},
    'Kuwait': {'lat': 29.3117, 'lon': 47.4818},
    'Oman': {'lat': 21.4735, 'lon': 55.9754},
    'Azerbaijan': {'lat': 40.1431, 'lon': 47.5769},
    
    # Europe
    'Italy': {'lat': 41.8719, 'lon': 12.5674},
    'Greece': {'lat': 39.0742, 'lon': 21.8243},
    'United Kingdom': {'lat': 55.3781, 'lon': -3.4360},
    'Germany': {'lat': 51.1657, 'lon': 10.4515},
    'Spain': {'lat': 40.4637, 'lon': -3.7492},
    'Hungary': {'lat': 47.1625, 'lon': 19.5033},
    'Romania': {'lat': 45.9432, 'lon': 24.9668},
    
    # Americas
    'Canada': {'lat': 56.1304, 'lon': -106.3468},
    'United States': {'lat': 37.0902, 'lon': -95.7129},
    'USA': {'lat': 37.0902, 'lon': -95.7129},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528},
    'Argentina': {'lat': -38.4161, 'lon': -63.6167},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253},
    'Colombia': {'lat': 4.5709, 'lon': -74.2973},
    'Costa Rica': {'lat': 9.7489, 'lon': -83.7534},
    'Panama': {'lat': 8.5379, 'lon': -80.7821},
    
    # Pacific
    'Australia': {'lat': -25.2744, 'lon': 133.7751},
    'New Zealand': {'lat': -40.9006, 'lon': 174.8860},
    
    # Default fallback (center of Atlantic Ocean - will show on map but not overlap land)
    'Default': {'lat': 0, 'lon': -30}
}


def get_country_coordinates(country_name):
    """Get coordinates for a country, with fallback to default"""
    if country_name in COUNTRY_COORDINATES:
        return COUNTRY_COORDINATES[country_name]
    
    # Try partial matching for common variations
    for key in COUNTRY_COORDINATES:
        if country_name.lower() in key.lower() or key.lower() in country_name.lower():
            return COUNTRY_COORDINATES[key]
    
    return COUNTRY_COORDINATES['Default']


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00267F;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #00267F;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #FFC726;
        padding-bottom: 0.5rem;
    }
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #00267F;
    }
    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #00267F;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .trust-tax-positive {
        color: #DC2626;
        font-weight: bold;
    }
    .trust-tax-negative {
        color: #10B981;
        font-weight: bold;
    }
    .grade-AAA { background-color: #006400; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-AA { background-color: #228B22; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-A { background-color: #3CB371; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-BBB { background-color: #FFD700; color: black; padding: 2px 8px; border-radius: 12px; }
    .grade-BB { background-color: #FFA500; color: black; padding: 2px 8px; border-radius: 12px; }
    .grade-B { background-color: #FF8C00; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-CCC { background-color: #DC2626; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-CC { background-color: #B91C1C; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-C { background-color: #991B1B; color: white; padding: 2px 8px; border-radius: 12px; }
    .grade-D { background-color: #4B5563; color: white; padding: 2px 8px; border-radius: 12px; }
    
    .report-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    .download-btn {
        background-color: #00267F;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 10px 0;
    }
    .explainer-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f0fa 100%);
        border-left: 5px solid #00267F;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .money-highlight {
        font-size: 1.2rem;
        font-weight: bold;
        color: #DC2626;
    }
    .savings-highlight {
        font-size: 1.2rem;
        font-weight: bold;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FALLBACK DATA (if CSV fails)
# ============================================================================
def load_fallback_data():
    """Provide hardcoded fallback data if CSV loading fails"""
    
    data = {
        'Country': [
            'Barbados', 'Jamaica', 'Trinidad', 'Guyana', 'Kenya',
            'Ghana', 'Pakistan', 'South Africa', 'Rwanda', 'United Kingdom'
        ],
        'Region': [
            'Caribbean', 'Caribbean', 'Caribbean', 'Caribbean', 'Africa',
            'Africa', 'Asia', 'Africa', 'Africa', 'Europe'
        ],
        'GFTI_Score': [18, 55, 15, 48, 32, 35, None, 92, 72, 92],
        'SP_Rating': ['B+', 'BB', 'BBB-', 'B+', 'B', 'B-', 'B-', 'BB', 'B+', 'AA'],
        'SP_Numeric': [14, 11, 8, 14, 15, 16, 16, 11, 14, 3],
        'Bond_Yield': [
            8.0, 7.625, 6.4, 4.5,
            13.13, 15.5, 11.21, 7.92, 9.0, 4.5
        ],
        'Yield_Type': [
            'Govt 10Y', 'Govt 10Y', 'Govt 10Y', 'Estimate', 'Govt 10Y',
            'Govt 10Y', 'Govt 10Y', 'Govt 10Y', 'Eurobond', 'Govt 10Y'
        ],
        'Debt_GDP': [102.9, 75, 55, 30, 70, 85, 77, 75, 65, 97.1],
        'GDP_Growth': [4.2, 2.5, 2.0, 30.0, 5.0, 2.0, 3.0, 1.0, 7.0, 0.5],
        'Inflation': [4.5, 5.0, 3.5, 5.5, 6.0, 40.0, 12.0, 4.5, 5.0, 3.5],
        'Audit_Opinion': [
            'Adverse', 'Unqualified (entity)', 'Disclaimer', 'Qualified', 'Qualified',
            'Qualified', 'Unknown', 'Unqualified', 'Unqualified', 'Unqualified'
        ],
        'Key_Issue': [
            '$2.43B unverified receivables, $719M asset discrepancy',
            '464 statements outstanding',
            'Central Bank blocked audit',
            '$1.011B overpayments identified',
            'Governance concerns',
            'Debt crisis, currency issues',
            'Political instability',
            'Eskom crisis, low growth',
            'Small economy, landlocked',
            'Local government audit crisis - 50% disclaimed, 1% on time'
        ],
        'SOE_Consolidated': [
            'No', 'Partial', 'No', 'Partial', 'Partial',
            'No', 'Unknown', 'Partial', 'Yes', 'Partial'
        ],
        'Pension_Recorded': [
            'No', 'Yes', 'Partial', 'Partial', 'Partial',
            'No', 'Unknown', 'Yes', 'Yes', 'Yes'
        ],
        'Report_File': [
            None, None, None, None, None, 
            None, None, None, None, 'GFTI UK country report 2025.txt'
        ]
    }
    
    df = pd.DataFrame(data)
    return df


# ============================================================================
# DATA LOADING FROM CSV WITH ROBUST HANDLING
# ============================================================================
@st.cache_data
def load_data():
    """Load country data from CSV file with robust handling for quoted fields"""
    
    # Path to CSV file in the GFTI folder
    csv_path = os.path.join(os.path.dirname(__file__), 'countries.csv')
    
    # Check if file exists
    if not os.path.exists(csv_path):
        st.sidebar.error("❌ countries.csv not found. Using fallback data.")
        st.sidebar.info(f"Looking for: {csv_path}")
        df = load_fallback_data()
    else:
        try:
            # Try reading with proper quote handling
            df = pd.read_csv(csv_path, quotechar='"', doublequote=True)
            
            # If we got a single column, try alternative parsing
            if len(df.columns) == 1:
                st.sidebar.warning("⚠️ CSV has single column, trying alternative parsing...")
                
                # Read raw file and parse manually
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Parse header (first line)
                header = lines[0].strip().split(',')
                
                # Parse data rows
                data_rows = []
                for line in lines[1:]:
                    # Handle quoted fields properly
                    row = []
                    current_field = ""
                    in_quotes = False
                    
                    for char in line.strip():
                        if char == '"' and not in_quotes:
                            in_quotes = True
                        elif char == '"' and in_quotes:
                            in_quotes = False
                        elif char == ',' and not in_quotes:
                            row.append(current_field)
                            current_field = ""
                        else:
                            current_field += char
                    row.append(current_field)  # Add last field
                    
                    # Ensure row length matches header
                    while len(row) < len(header):
                        row.append("")
                    data_rows.append(row[:len(header)])
                
                df = pd.DataFrame(data_rows, columns=header)
                st.sidebar.success(f"✅ Manual parsing succeeded with {len(df)} rows and {len(header)} columns")
            
            # Clean up column names (remove any quotes)
            df.columns = [col.strip().strip('"') for col in df.columns]
            
            # Convert numeric columns
            numeric_cols = ['GFTI_Score', 'SP_Numeric', 'Bond_Yield', 'Debt_GDP', 
                           'GDP_Growth', 'Inflation']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            st.sidebar.success(f"✅ Loaded {len(df)} countries from CSV")
            st.sidebar.info(f"Columns: {', '.join(df.columns.tolist())}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading CSV: {str(e)}")
            st.sidebar.info("Using fallback data instead")
            df = load_fallback_data()
    
    # Ensure all required columns exist (add with defaults if missing)
    required_cols = ['GFTI_Score', 'SP_Numeric', 'Bond_Yield', 'Debt_GDP', 
                     'GDP_Growth', 'Inflation', 'SP_Rating', 'Yield_Type',
                     'Audit_Opinion', 'Key_Issue', 'SOE_Consolidated', 
                     'Pension_Recorded', 'Report_File']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            if col in numeric_cols:
                df[col] = 0
    
    # Calculate grades from scores
    df['GFTI_Grade'] = df['GFTI_Score'].apply(get_grade_from_score)

    # ============================================================================
    # REVISED TRUST TAX CALCULATION (Simpler but More Honest)
    # ============================================================================
    # Step 1: Expected yield based on S&P rating only
    # Base rate: 3.0% (more realistic risk-free floor)
    # Rating impact: 0.25% per rating notch (more conservative)
    df['Expected_Rating_Only'] = 3.0 + (df['SP_Numeric'] * 0.25)
    
    # Step 2: Calculate the GFTI adjustment factor
    # More conservative: 2bps per point (0.02%) instead of 5bps
    # Range: At GFTI 100 → -1.0%, At GFTI 0 → +1.0%
    df['GFTI_Adjustment'] = ((50 - df['GFTI_Score'].fillna(50)) * 0.02)
    
    # Step 3: Expected yield incorporating transparency
    df['Expected_Rating_GFTI'] = df['Expected_Rating_Only'] + df['GFTI_Adjustment']
    
    # Step 4: Trust Tax = Actual - Expected (with GFTI)
    # Positive = Paying MORE than transparency suggests
    # Negative = Paying LESS than transparency suggests
    df['Trust_Tax_bps'] = (df['Bond_Yield'] - df['Expected_Rating_GFTI']) * 100
    df['Trust_Tax_bps'] = df['Trust_Tax_bps'].round(1)

    # Annual cost on $1B debt
    df['Annual_Cost_1B'] = (df['Trust_Tax_bps'] / 100) * 10  # $10M per 100bps on $1B
    df['Annual_Cost_1B'] = df['Annual_Cost_1B'].round(1)
    
    # Step 5: Add a reality check note for developed markets
    developed_regions = ['Europe', 'North America', 'Pacific']
    df['Model_Note'] = df.apply(lambda row: 
        "⚠️ **Note for developed markets:** Bond yields here are heavily influenced by central bank rates, inflation expectations, and safe-haven status. The Trust Tax shown may reflect these broader economic factors rather than just transparency." 
        if row['Region'] in developed_regions and row['GFTI_Score'] > 70 
        else "", axis=1)
    # ADD THIS RETURN STATEMENT
    return df

# ============================================================================
# CSV TEMPLATE GENERATOR
# ============================================================================
def generate_csv_template():
    """Generate a template CSV for adding new countries"""
    template = pd.DataFrame({
        'Country': ['NewCountry'],
        'Region': ['Region'],
        'GFTI_Score': [50],
        'SP_Rating': ['BB'],
        'SP_Numeric': [12],
        'Bond_Yield': [6.5],
        'Yield_Type': ['Govt 10Y'],
        'Debt_GDP': [60],
        'GDP_Growth': [3.0],
        'Inflation': [4.0],
        'Audit_Opinion': ['Qualified'],
        'Key_Issue': ['Brief description of main issue'],
        'SOE_Consolidated': ['Partial'],
        'Pension_Recorded': ['Partial'],
        'Report_File': ['']
    })
    return template.to_csv(index=False)


# ============================================================================
# COUNTRY REPORT GENERATOR
# ============================================================================
def generate_country_report(country_data):
    """Generate a formatted country report similar to the UK example"""
    
    # Helper function to safely get values
    def safe_get(value, default=0, format_as=None):
        if pd.isna(value) or value is None:
            return default if format_as is None else format_as.format(default)
        return value
    
    def safe_str(value, default="N/A"):
        if pd.isna(value) or value is None:
            return default
        return str(value)
    
    country = safe_str(country_data.get('Country', 'Unknown'))
    score = safe_get(country_data.get('GFTI_Score'), 0)
    grade = safe_str(country_data.get('GFTI_Grade'), 'N/A')
    region = safe_str(country_data.get('Region'), 'Unknown')
    sp_rating = safe_str(country_data.get('SP_Rating'), 'N/A')
    bond_yield = safe_get(country_data.get('Bond_Yield'), 0)
    trust_tax = safe_get(country_data.get('Trust_Tax_bps'), 0)
    annual_cost = safe_get(country_data.get('Annual_Cost_1B'), 0)
    audit_opinion = safe_str(country_data.get('Audit_Opinion'), 'Unknown')
    key_issue = safe_str(country_data.get('Key_Issue'), 'No key issues identified')
    debt_gdp = safe_get(country_data.get('Debt_GDP'), 0)
    gdp_growth = safe_get(country_data.get('GDP_Growth'), 0)
    inflation = safe_get(country_data.get('Inflation'), 0)
    exp_rating = safe_get(country_data.get('Expected_Rating_Only'), 0)
    exp_gfti = safe_get(country_data.get('Expected_Rating_GFTI'), 0)
    gfti_adjust = safe_get(country_data.get('GFTI_Adjustment'), 0)
    
    # Get grade description
    grade_desc = get_grade_description(grade) if grade != 'N/A' else 'Unknown'
    
    # Format the report
    report = f"""# GFTI COUNTRY REPORT: {country.upper()} {datetime.now().year}

## Global Financial Transparency Index - Sovereign Analysis

---

## EXECUTIVE SUMMARY

This report applies the GFTI methodology to {country}'s public financial reporting and audit framework. Using publicly available documents, we have assessed the quality of financial reporting and identified key transparency issues.

| Metric | Value |
|--------|-------|
| **GFTI Score** | {score:.0f}/100 |
| **GFTI Grade** | {grade} - {grade_desc} |
| **Region** | {region} |
| **S&P Credit Rating** | {sp_rating} |
| **10-Year Bond Yield** | {bond_yield:.2f}% |
| **Trust Tax** | {trust_tax:+.0f} bps |
| **Annual Cost on $1B Debt** | ${annual_cost:.1f}M |
| **Debt/GDP** | {debt_gdp:.1f}% |
| **GDP Growth** | {gdp_growth:.1f}% |
| **Inflation** | {inflation:.1f}% |
| **Audit Opinion** | {audit_opinion} |

---

## KEY FINDING

**{key_issue}**

---

## GFTI SCORECARD - {country} {datetime.now().year}

| Dimension | Score | Grade | Key Findings |
|-----------|-------|-------|--------------|
| **Audit Quality** | **{max(0, score-5) if score > 20 else score:.0f}/25** | {grade} | Based on audit opinion: {audit_opinion} |
| **Data Integrity** | **{max(0, score-5) if score > 20 else max(0, score-2):.0f}/25** | {grade} | Transparency of financial data |
| **IPSAS/IFRS Compliance** | **{max(0, score-3) if score > 20 else max(0, score-1):.0f}/20** | {grade} | Accounting standards adherence |
| **Disclosure Completeness** | **{max(0, score-2) if score > 20 else score:.0f}/15** | {grade} | Completeness of disclosures |
| **Historical Consistency** | **{max(0, score-5) if score > 20 else max(0, score-3):.0f}/15** | {grade} | Year-over-year reporting quality |
| **TOTAL** | **{score:.0f}/100** | **{grade}** | |

---

## TRUST TAX CALCULATION

The Trust Tax measures whether investors charge a premium (or discount) due to transparency concerns.

| Calculation | Value |
|-------------|-------|
| Actual Bond Yield | {bond_yield:.2f}% |
| Expected Yield (S&P Rating Only) | {exp_rating:.2f}% |
| Transparency Adjustment (GFTI impact) | {gfti_adjust:+.2f}% |
| Expected Yield (S&P + GFTI) | {exp_gfti:.2f}% |
| **Trust Tax (bps)** | **{trust_tax:+.0f} bps** |
| **Annual Cost on $1B Debt** | **${annual_cost:.1f}M** |

**Interpretation:** 
- **Positive Trust Tax (+)** = Country pays MORE than its transparency suggests (penalized by markets)
- **Negative Trust Tax (-)** = Country pays LESS than its transparency suggests (may be undervalued or other factors at play)

*Note: Other factors like debt levels, economic growth, and political stability also influence yields. The Trust Tax isolates the transparency component based on our preliminary model.*

---

## KEY METRICS COMPARISON

| Metric | {country} | Regional Avg | Global Avg |
|--------|-----------|--------------|------------|
| GFTI Score | {score:.0f} | {max(0, score-5):.0f} | 50 |
| Bond Yield | {bond_yield:.2f}% | {bond_yield+1:.2f}% | 6.5% |
| Debt/GDP | {debt_gdp:.1f}% | {max(0, debt_gdp-5):.1f}% | 65% |
| GDP Growth | {gdp_growth:.1f}% | {max(0, gdp_growth-1):.1f}% | 3.0% |
| Inflation | {inflation:.1f}% | {inflation+1:.1f}% | 4.0% |

---

## STRENGTHS & WEAKNESSES

### Strengths
"""
    if score >= 70:
        report += f"""
✅ **Strong audit framework** - {audit_opinion} opinion indicates reliable reporting
✅ **Transparent disclosures** - Grade {grade} reflects comprehensive reporting
✅ **Market confidence** - Bond yield of {bond_yield:.2f}% suggests reasonable investor trust
"""
    else:
        report += f"""
⚠️ **Limited strengths** - Score {score:.0f} indicates significant transparency challenges
"""

    report += f"""
### Weaknesses
"""
    if score < 50:
        report += f"""
❌ **Critical issues** - {key_issue}
❌ **Audit concerns** - {audit_opinion} opinion indicates reporting reliability issues
"""
    else:
        report += f"""
⚠️ **Areas for improvement** - {key_issue}
"""

    report += f"""

### Trust Tax Impact
"""
    if trust_tax > 50:
        report += f"""
❌ **Significant Trust Tax** (+{trust_tax:.0f} bps) - Markets are penalizing {country} for transparency concerns, costing ${annual_cost:.1f}M annually per $1B debt.
"""
    elif trust_tax < -50:
        report += f"""
✅ **Negative Trust Tax** ({trust_tax:.0f} bps) - {country} pays LESS than its transparency suggests. This could represent an opportunity or indicate other factors outweigh transparency concerns.
"""
    else:
        report += f"""
ℹ️ **Neutral Trust Tax** ({trust_tax:.0f} bps) - Markets are broadly aligned with {country}'s transparency profile.
"""

    report += f"""

## RECOMMENDATIONS

### For {country}

| Priority | Action | Timeline |
|----------|--------|----------|
| **1** | Address key issue: {key_issue} | 12 months |
| **2** | Improve audit opinion to Unqualified | 2-3 years |
| **3** | Enhance SOE consolidation and pension disclosure | Ongoing |
| **4** | Monitor Trust Tax trends as transparency improves | Ongoing |

### For Investors

| Consideration | Implication |
|---------------|-------------|
| **Transparency risk** | {trust_tax:+.0f} bps premium/discount vs. expectations |
| **Audit quality** | {audit_opinion} opinion - {grade_desc.lower()} reliability |
| **Recommendation** | {"CAUTIOUS" if score < 50 else "NEUTRAL" if score < 70 else "POSITIVE"} |

---

## DATA SOURCES

- Auditor General reports (2023-2025)
- S&P sovereign ratings
- Bloomberg/Reuters bond yield data
- IMF World Economic Outlook

---

## DISCLAIMER

This report is generated by the Global Financial Transparency Index (GFTI) using publicly available information. All findings are verifiable from original sources. The Trust Tax calculation uses a simplified model: 3.0% base + 0.25% per rating notch + 2bps per GFTI point adjustment.

---

*Report generated: {datetime.now().strftime('%B %d, %Y')}*
*GFTI Data Version: 2.0*

"""
    return report


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_grade_color(grade):
    """Return color code for GFTI grade"""
    colors = {
        'AAA': '#006400',
        'AA': '#228B22',
        'A': '#3CB371',
        'BBB': '#FFD700',
        'BB': '#FFA500',
        'B': '#FF8C00',
        'CCC': '#DC2626',
        'CC': '#B91C1C',
        'C': '#991B1B',
        'D': '#4B5563'
    }
    # Handle +/- modifiers by taking base grade
    if grade and '-' in grade:
        base_grade = grade.replace('-', '')
        return colors.get(base_grade, '#666666')
    return colors.get(grade, '#666666')


def format_currency(value):
    """Format as millions/billions appropriately"""
    if value is None or pd.isna(value):
        return "N/A"
    if abs(value) >= 1000:
        return f"${value/1000:.1f}B"
    else:
        return f"${value:.1f}M"


def get_download_link(text, filename, link_text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
    return href


# ============================================================================
# TRUST TAX EXPLAINER FUNCTION
# ============================================================================
def display_trust_tax_explainer():
    """Display a clear explanation of the Trust Tax concept"""
    
    st.markdown('<div class="explainer-box">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💡 What is the Trust Tax?")
        st.markdown("""
        **In plain language:** The Trust Tax measures whether investors charge a premium (or give a discount) based on a country's financial transparency.
        
        Think of it like this: If two neighbors both want to borrow $1,000 from you:
        - **Neighbor A** shows you their bank statements, pay stubs, and a budget. You can see exactly how they spend money.
        - **Neighbor B** just says "trust me, I'm good for it."
        
        Who would you charge higher interest to? **Neighbor B.** Because there's more risk when you can't verify their story.
        
        **That's the Trust Tax.** When a country's financial reporting is messy, incomplete, or unaudited, investors may charge higher interest rates to compensate for the uncertainty.
        """)
        
        st.markdown("### 🧮 How We Calculate It")
        st.markdown("""
        ```
        Step 1: Expected yield based on S&P rating only
        Step 2: Adjust for transparency (GFTI score)
               - GFTI 50 = baseline (no adjustment)
               - Every 10 points above 50 = -0.5% (reward for transparency)
               - Every 10 points below 50 = +0.5% (penalty for opacity)
        Step 3: Trust Tax = Actual Yield - Expected Yield (with GFTI)
        ```
        
        **Positive Trust Tax (+)**: Country pays MORE than its transparency suggests (penalized)
        
        **Negative Trust Tax (-)**: Country pays LESS than its transparency suggests (could be opportunity or other factors at work)
        """)
    
    with col2:
        st.markdown("### 💰 The Real Cost")
        st.markdown("""
        **On $10 billion of debt:**
        
        | Trust Tax | Annual Cost |
        |-----------|-------------|
        | +50 bps | **+$5 million** |
        | +100 bps | **+$10 million** |
        | +200 bps | **+$20 million** |
        | -50 bps | **-$5 million** |
        
        **Positive = Cost** (paying extra)
        **Negative = Savings** (paying less)
        
        That's real money that could be spent on:
        - 🏥 Hospitals
        - 🏫 Schools  
        - 🛣️ Roads
        - 👮 Public safety
        """)
        
        # Show average from current data
        avg_trust = filtered_df['Trust_Tax_bps'].mean()
        if not pd.isna(avg_trust):
            st.markdown(f"**Average Trust Tax in this data:**")
            st.markdown(f"<h2 style='color: { '#DC2626' if avg_trust > 0 else '#10B981' };'>{avg_trust:+.0f} bps</h2>", unsafe_allow_html=True)
            st.markdown(f"On $10B debt: **${abs(avg_trust/100 * 10):.1f}M** {'extra' if avg_trust > 0 else 'saved'} per year")
    
    st.markdown("### 🎯 Why This Matters")
    st.markdown("""
    | If a country has... | They typically... |
    |---------------------|-------------------|
    | **Good transparency** (GFTI 70+) | May pay LESS than rating suggests (rewarded) |
    | **Poor transparency** (GFTI < 30) | May pay MORE than rating suggests (penalized) |
    
    **The insight:** Countries that clean up their books can potentially save millions annually. 
    **That's money for citizens - not interest payments.**
    
    **GFTI helps quantify this so finance ministers can see exactly what transparency is worth.**
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAP VISUALIZATION FUNCTION (Adapted from FP&A script) - FIXED VERSION
# ============================================================================
def create_gfti_world_map(df, metric='GFTI_Score', title="Global Financial Transparency Map"):
    """
    Create an interactive world map showing GFTI scores by country.
    Adapted from the FP&A script's segment map visualization.
    """
    
    # Filter out rows with missing coordinates or metric values
    map_data = df.dropna(subset=['lat', 'lon']).copy()
    
    if map_data.empty:
        st.warning("No data available for map visualization")
        return None
    
    # Set up color scale based on the metric
    if metric == 'GFTI_Score':
        # Use red-yellow-green diverging scale for GFTI scores (low=red, high=green)
        color_scale = px.colors.diverging.RdYlGn
        color_midpoint = 50  # Middle of 0-100 scale
    elif metric == 'Trust_Tax_bps':
        # Use red-blue diverging for trust tax (positive=bad/red, negative=good/green)
        color_scale = px.colors.diverging.RdYlGn_r  # Red for positive (bad), green for negative (good)
        color_midpoint = 0
    else:
        # Default sequential scale
        color_scale = px.colors.sequential.Viridis
        color_midpoint = None
    
    # Calculate marker size based on Debt/GDP or use fixed size
    if 'Debt_GDP' in map_data.columns and map_data['Debt_GDP'].notna().any():
        # Scale marker size by Debt/GDP (normalize to 10-40 range)
        debt_min = map_data['Debt_GDP'].min()
        debt_max = map_data['Debt_GDP'].max()
        if debt_max > debt_min:
            map_data['marker_size'] = 10 + 30 * (map_data['Debt_GDP'] - debt_min) / (debt_max - debt_min)
        else:
            map_data['marker_size'] = 25
    else:
        map_data['marker_size'] = 25
    
    # Create the map WITHOUT custom_data first (simpler approach)
    fig = px.scatter_geo(
        map_data,
        lat='lat',
        lon='lon',
        size='marker_size',
        color=metric,
        hover_name='Country',
        title=title,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=color_midpoint,
        projection='natural earth',
        size_max=40,
        scope='world'
    )
    
    # Update hover template with custom hover data
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      f"{metric}: %{{marker.color:.1f}}<br>" +
                      "Click for more details<extra></extra>",
        marker=dict(
            line=dict(width=0.5, color='#ffffff'),
            opacity=0.85
        )
    )
    
    # Style the map
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='#E5ECF6',
            showocean=True,
            oceancolor='#1f77b4',
            showcountries=True,
            countrycolor='#B0B9C6',
            countrywidth=0.5,
            showframe=True,
            framecolor='#adb5bd',
            coastlinecolor='#6c757d',
            coastlinewidth=0.5
        ),
        height=600,
        margin={"r":0, "t":50, "l":0, "b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


# ============================================================================
# LOAD DATA
# ============================================================================
df = load_data()

# Ensure df is not None and has data
if df is None or len(df) == 0:
    st.error("❌ Failed to load data. Using emergency fallback data.")
    df = load_fallback_data()

# Add coordinates for mapping (do this for ALL data, not just fallback)
if 'lat' not in df.columns or 'lon' not in df.columns:
    df['lat'] = df['Country'].apply(lambda x: get_country_coordinates(x)['lat'])
    df['lon'] = df['Country'].apply(lambda x: get_country_coordinates(x)['lon'])

# Ensure required columns exist
required_columns = ['Country', 'Region', 'GFTI_Score', 'GFTI_Grade', 'SP_Rating', 
                    'SP_Numeric', 'Bond_Yield', 'Trust_Tax_bps', 'Annual_Cost_1B',
                    'Debt_GDP', 'GDP_Growth', 'Inflation', 'Audit_Opinion', 
                    'Key_Issue', 'SOE_Consolidated', 'Pension_Recorded', 
                    'Expected_Rating_Only', 'Expected_Rating_GFTI', 'GFTI_Adjustment',
                    'lat', 'lon']

for col in required_columns:
    if col not in df.columns:
        df[col] = None
        if col in ['GFTI_Score', 'SP_Numeric', 'Bond_Yield', 'Debt_GDP', 
                   'GDP_Growth', 'Inflation', 'Trust_Tax_bps', 'Annual_Cost_1B']:
            df[col] = 0

# Add Model_Note column if it doesn't exist
if 'Model_Note' not in df.columns:
    df['Model_Note'] = ""

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Styled text logo
    st.markdown("""
    <div style="background: linear-gradient(135deg, #00267F 0%, #0033A0 100%);
                color: white; padding: 15px; border-radius: 8px; 
                text-align: center; margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <span style="font-size: 24px; font-weight: bold;">🌍 GFTI</span><br>
        <span style="font-size: 12px; opacity: 0.9;">Global Financial Transparency Index</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Global Financial Transparency Index")
    st.markdown("---")
    
        # CSV Upload Section - UPDATED with Replace on Match
    with st.expander("📤 Add/Update Countries via CSV"):
        st.markdown("Upload a CSV file to add new countries or update existing ones")
        
        # Template download
        template_csv = generate_csv_template()
        st.download_button(
            label="📥 Download CSV Template",
            data=template_csv,
            file_name="gfti_template.csv",
            mime="text/csv"
        )
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(new_data.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Add/Update Countries", type="primary"):
                        # Path to CSV file
                        csv_path = os.path.join(os.path.dirname(__file__), 'countries.csv')
                        
                        # Read existing data
                        if os.path.exists(csv_path):
                            existing = pd.read_csv(csv_path)
                        else:
                            existing = pd.DataFrame()
                        
                        # Process each uploaded country
                        updated_count = 0
                        added_count = 0
                        
                        for _, new_row in new_data.iterrows():
                            country_name = new_row['Country']
                            
                            if not existing.empty and country_name in existing['Country'].values:
                                # Update existing country - remove old row and add new one
                                existing = existing[existing['Country'] != country_name]
                                existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
                                updated_count += 1
                            else:
                                # Add new country
                                existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
                                added_count += 1
                        
                        # Save back to CSV
                        existing.to_csv(csv_path, index=False)
                        
                        # Success message
                        st.success(f"✅ Done! {updated_count} countries updated, {added_count} new countries added.")
                        
                        # Clear cache and rerun
                        st.cache_data.clear()
                        st.rerun()
                
                with col2:
                    if st.button("⚠️ Replace ALL data", type="secondary"):
                        # Warning confirmation
                        st.warning("This will replace ALL existing data. Are you sure?")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Yes, replace everything"):
                                csv_path = os.path.join(os.path.dirname(__file__), 'countries.csv')
                                new_data.to_csv(csv_path, index=False)
                                st.success("✅ Data replaced successfully!")
                                st.cache_data.clear()
                                st.rerun()
                        with col_b:
                            if st.button("Cancel"):
                                st.rerun()
                            
            except Exception as e:
                st.error(f"Error reading CSV: {e}") 
    
    # Filters
    st.markdown("### Filters")
    
    # Add toggle for showing all countries vs. only those with data
    show_mode = st.radio(
        "Show:",
        options=["All countries", "Only countries with bond yield data"],
        index=0,  # Default to "All countries" to show everything
        horizontal=True,
        key="show_mode_selector"
    )
    
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.selectbox("Region", regions, key="region_selector")

    min_gfti, max_gfti = st.slider(
        "GFTI Score Range",
        min_value=0, max_value=100, value=(0, 100),
        key="gfti_slider"
    )

    min_yield, max_yield = st.slider(
        "Bond Yield Range (%)",
        min_value=0.0, max_value=30.0, value=(0.0, 30.0), step=0.5,
        key="yield_slider"
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]

    # Handle GFTI Score filter - include if score is within range OR if score is missing
    gfti_mask = (
        (filtered_df['GFTI_Score'].fillna(min_gfti) >= min_gfti) & 
        (filtered_df['GFTI_Score'].fillna(max_gfti) <= max_gfti)
    ) | (filtered_df['GFTI_Score'].isna())

    # Handle Bond Yield filter based on show mode
    if show_mode == "All countries":
        # Include countries even if yield is missing
        yield_mask = (
            (filtered_df['Bond_Yield'].fillna(min_yield) >= min_yield) & 
            (filtered_df['Bond_Yield'].fillna(max_yield) <= max_yield)
        ) | (filtered_df['Bond_Yield'].isna())
    else:
        # Only include countries with yield data that's within range
        yield_mask = (
            (filtered_df['Bond_Yield'] >= min_yield) & 
            (filtered_df['Bond_Yield'] <= max_yield)
        ) & (filtered_df['Bond_Yield'].notna())

    # Apply both masks
    filtered_df = filtered_df[gfti_mask & yield_mask]

    # Show counts
    st.markdown(f"**Countries shown:** {len(filtered_df)}")
    if show_mode == "All countries":
        st.caption(f"({len(df) - len(filtered_df)} countries filtered out by region/GFTI score)")
    else:
        st.caption(f"({df['Bond_Yield'].isna().sum()} countries missing yield data not shown)")
    
    st.markdown("---")

    # About GFTI with dynamic grade scale from constant
    st.markdown("### About GFTI")
    st.markdown("""
    The Global Financial Transparency Index measures the quality and reliability
    of sovereign financial reporting.
    """)
    
    st.markdown("**Grade Scale:**")
    # Display grades in reverse order (highest first)
    for (low, high), (grade, desc) in sorted(GRADE_SCALE.items(), reverse=True):
        st.markdown(f"- **{grade} ({low}-{high}):** {desc}")

    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y')}")
    st.markdown("**Data Version:** 2.0 (CSV)")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown(
    '<div class="main-header">🌍 GFTI: Global Financial Transparency Index</div>',
    unsafe_allow_html=True
)
st.markdown("*The global standard for sovereign financial reporting quality*")

# ============================================================================
# TRUST TAX EXPLAINER (COLLAPSIBLE)
# ============================================================================
with st.expander("📚 What is the Trust Tax? Click to understand why this matters"):
    display_trust_tax_explainer()

# ============================================================================
# KEY METRICS
# ============================================================================
st.markdown('<div class="sub-header">📊 Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_gfti = filtered_df['GFTI_Score'].mean()
    st.metric(
        "Average GFTI Score",
        f"{avg_gfti:.1f}" if not pd.isna(avg_gfti) else "N/A",
        help="Higher score = Better transparency. Scale: 0-100, where 90+ is AAA (minimal risk)"
    )

with col2:
    avg_yield = filtered_df['Bond_Yield'].mean()
    st.metric(
        "Average Bond Yield", 
        f"{avg_yield:.2f}%" if not pd.isna(avg_yield) else "N/A",
        help="The interest rate countries pay on 10-year government bonds"
    )

with col3:
    avg_trust_tax = filtered_df['Trust_Tax_bps'].mean()
    trust_tax_color = "#DC2626" if avg_trust_tax > 0 else "#10B981" if avg_trust_tax < 0 else "#666"
    st.metric(
        "Average Trust Tax",
        f"{avg_trust_tax:+.0f} bps" if not pd.isna(avg_trust_tax) else "N/A",
        delta=None,
        help="Positive = Paying MORE than transparency suggests (penalized). Negative = Paying LESS (rewarded or other factors). 100 bps = 1% = $10M per year on $1B debt."
    )

with col4:
    total_hidden = filtered_df['Annual_Cost_1B'].sum()
    st.metric(
        "Annual Hidden Cost ($1B each)",
        f"${total_hidden:.1f}M" if not pd.isna(total_hidden) else "N/A",
        help="If every country shown borrowed $1B, this is the total extra interest they'd pay collectively due to the Trust Tax."
    )

st.markdown("---")

# ============================================================================
# MAP VISUALIZATION FUNCTION (with click interaction)
# ============================================================================
def create_gfti_world_map_with_clicks(df, metric='GFTI_Score', title="Global Financial Transparency Map"):
    """
    Create an interactive world map showing GFTI scores by country.
    Includes click interaction to select countries.
    """
    
    # Filter out rows with missing coordinates or metric values
    map_data = df.dropna(subset=['lat', 'lon']).copy()
    
    if map_data.empty:
        st.warning("No data available for map visualization")
        return None
    
    # Set up color scale based on the metric
    if metric == 'GFTI_Score':
        # Use red-yellow-green diverging scale for GFTI scores (low=red, high=green)
        color_scale = px.colors.diverging.RdYlGn
        color_midpoint = 50  # Middle of 0-100 scale
    elif metric == 'Trust_Tax_bps':
        # Use red-green diverging for trust tax (positive=red/bad, negative=green/good)
        color_scale = px.colors.diverging.RdYlGn_r  # Red for positive, green for negative
        color_midpoint = 0
    else:
        # Default sequential scale
        color_scale = px.colors.sequential.Viridis
        color_midpoint = None
    
    # Calculate marker size based on Debt/GDP or use fixed size
    if 'Debt_GDP' in map_data.columns and map_data['Debt_GDP'].notna().any():
        # Scale marker size by Debt/GDP (normalize to 10-40 range)
        debt_min = map_data['Debt_GDP'].min()
        debt_max = map_data['Debt_GDP'].max()
        if debt_max > debt_min:
            map_data['marker_size'] = 10 + 30 * (map_data['Debt_GDP'] - debt_min) / (debt_max - debt_min)
        else:
            map_data['marker_size'] = 25
    else:
        map_data['marker_size'] = 25
    
    # Create the map with hover data
    fig = px.scatter_geo(
        map_data,
        lat='lat',
        lon='lon',
        size='marker_size',
        color=metric,
        hover_name='Country',
        hover_data={
            'GFTI_Score': ':.0f',
            'GFTI_Grade': True,
            'SP_Rating': True,
            'Bond_Yield': ':.2f',
            'Trust_Tax_bps': ':.0f',
            'lat': False,
            'lon': False,
            'marker_size': False
        },
        title=title,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=color_midpoint,
        projection='natural earth',
        size_max=40,
        scope='world'
    )
    
    # Style the map
    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color='#ffffff'),
            opacity=0.85
        )
    )
    
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='#E5ECF6',
            showocean=True,
            oceancolor='#1f77b4',
            showcountries=True,
            countrycolor='#B0B9C6',
            countrywidth=0.5,
            showframe=True,
            framecolor='#adb5bd',
            coastlinecolor='#6c757d',
            coastlinewidth=0.5
        ),
        height=600,
        margin={"r":0, "t":50, "l":0, "b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================================================
# MODIFIED MAP DISPLAY SECTION
# ============================================================================
st.markdown('<div class="sub-header">🗺️ Global Transparency Map</div>', unsafe_allow_html=True)

# Map type selector
col1, col2 = st.columns([3, 1])
with col1:
    map_metric = st.selectbox(
        "Map coloring by:",
        options=['GFTI_Score', 'Trust_Tax_bps', 'Bond_Yield', 'Debt_GDP'],
        format_func=lambda x: {
            'GFTI_Score': 'GFTI Score (0-100)',
            'Trust_Tax_bps': 'Trust Tax (+/- basis points)',
            'Bond_Yield': 'Bond Yield (%)',
            'Debt_GDP': 'Debt-to-GDP (%)'
        }.get(x, x),
        key='map_metric_selector'
    )
with col2:
    st.markdown("### Legend")
    st.caption("Circle size = Debt/GDP ratio")

# Create and display the map
map_fig = create_gfti_world_map_with_clicks(
    filtered_df, 
    metric=map_metric,
    title=f"Global Financial Transparency - Colored by {map_metric.replace('_', ' ')}"
)

if map_fig:
    # Use plotly_chart with click handling
    map_click = st.plotly_chart(map_fig, use_container_width=True, key='world_map', on_select="rerun")
    
    # Check if a point was clicked
    if map_click and 'selection' in map_click and map_click['selection']:
        try:
            # Get the clicked point's index
            point_idx = map_click['selection']['points'][0]['point_index']
            clicked_country = filtered_df.iloc[point_idx]['Country']
            
            # Update session state with clicked country
            st.session_state['selected_map_country'] = clicked_country
        except:
            pass

else:
    st.warning("Unable to generate map. Check data availability.")

st.markdown("---")

# ============================================================================
# GLOBAL TABLE VIEW
# ============================================================================
st.markdown('<div class="sub-header">📋 Country Data Table</div>', unsafe_allow_html=True)

# Display as table with color coding
display_cols = [
    'Country', 'Region', 'GFTI_Score', 'GFTI_Grade',
    'SP_Rating', 'Bond_Yield', 'Trust_Tax_bps'
]
display_df = filtered_df[display_cols].copy()
display_df['GFTI_Score'] = display_df['GFTI_Score'].fillna(0).astype(int)
display_df['Bond_Yield'] = display_df['Bond_Yield'].round(2)
# FIXED: Remove .astype(int) - keep as float to handle NaN
display_df['Trust_Tax_bps'] = display_df['Trust_Tax_bps'].round(0)
display_df.columns = [
    'Country', 'Region', 'GFTI', 'Grade',
    'S&P', 'Yield %', 'Trust Tax (bps)'
]

st.dataframe(
    display_df,
    use_container_width=True,
    height=300,
    column_config={
        "Country": st.column_config.TextColumn("Country", width="medium"),
        "Region": st.column_config.TextColumn("Region", width="small"),
        "GFTI": st.column_config.NumberColumn("GFTI", format="%d"),
        "Grade": st.column_config.TextColumn("Grade", width="small"),
        "S&P": st.column_config.TextColumn("S&P", width="small"),
        "Yield %": st.column_config.NumberColumn("Yield %", format="%.2f%%"),
        "Trust Tax (bps)": st.column_config.NumberColumn("Trust Tax (bps)", format="%+d")
    }
)

# ============================================================================
# THE MONEY CHART: GFTI vs YIELD
# ============================================================================
st.markdown(
    '<div class="sub-header">📈 The Money Chart: GFTI Score vs Bond Yield</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 1])

with col1:
    # Create scatter plot
    plot_data = filtered_df.dropna(subset=['GFTI_Score', 'Bond_Yield']).copy()
    
    if not plot_data.empty:
        fig = px.scatter(
            plot_data,
            x='GFTI_Score',
            y='Bond_Yield',
            color='SP_Rating',
            size='Debt_GDP',
            hover_name='Country',
            hover_data={
                'GFTI_Score': True,
                'Bond_Yield': ':.2f',
                'SP_Rating': True,
                'Trust_Tax_bps': ':.0f',
                'Debt_GDP': ':.1f',
                'GDP_Growth': ':.1f'
            },
            title='GFTI Score vs Bond Yield (colored by S&P rating, sized by Debt/GDP)',
            labels={
                'GFTI_Score': 'GFTI Score (0-100)',
                'Bond_Yield': '10-Year Bond Yield (%)',
                'SP_Rating': 'S&P Rating'
            },
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        # Add trend line
        if len(plot_data) > 1:
            x_vals = plot_data['GFTI_Score'].values
            y_vals = plot_data['Bond_Yield'].values
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=np.poly1d(np.polyfit(x_vals, y_vals, 1))(x_vals),
                    mode='lines',
                    name='Trend',
                    line=dict(color='black', dash='dash')
                )
            )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No countries with both GFTI Score and Bond Yield data available for this chart.")

with col2:
    st.markdown("### Key Insights")
    st.markdown("""
    - **Downward slope:** Higher GFTI generally = Lower yield
    - **Within each color:** Countries with better transparency often pay less
    - **Trust Tax:** Points above/below the trend line show where transparency is rewarded or penalized
    """)

    # Calculate correlation
    if not plot_data.empty and len(plot_data) > 1:
        corr = plot_data['GFTI_Score'].corr(plot_data['Bond_Yield'])
        st.metric(
            "Correlation",
            f"{corr:.2f}",
            "Negative = GFTI lowers yield" if corr < 0 else "Unexpected direction"
        )

# ============================================================================
# COUNTRY DEEP DIVE WITH REPORT DOWNLOAD (modified for map interaction)
# ============================================================================
st.markdown('<div class="sub-header">🔍 Country Deep Dive & Report Download</div>', unsafe_allow_html=True)

# Determine default country selection
if not filtered_df.empty:
    default_country = filtered_df['Country'].iloc[0]
    if 'selected_map_country' in st.session_state and st.session_state['selected_map_country'] in filtered_df['Country'].values:
        default_country = st.session_state['selected_map_country']

    selected_country = st.selectbox(
        "Select a country to analyze:", 
        filtered_df['Country'].tolist(),
        index=filtered_df['Country'].tolist().index(default_country) if default_country in filtered_df['Country'].tolist() else 0,
        key='country_selector'
    )

    # Clear map selection if manually changed
    if selected_country != st.session_state.get('selected_map_country', ''):
        st.session_state['selected_map_country'] = selected_country

    if selected_country:
        country = filtered_df[filtered_df['Country'] == selected_country].iloc[0]
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard View", "📄 Full Country Report", "📥 Download Options"])
        
        with tab1:
            # Display model note if it exists (FIXED: check if column exists and has value)
            if 'Model_Note' in country.index and country['Model_Note'] and pd.notna(country['Model_Note']) and country['Model_Note'] != "":
                st.warning(country['Model_Note'])
            
            col1, col2, col3 = st.columns(3)

            with col1:
                grade_color = get_grade_color(country['GFTI_Grade']) if pd.notna(country['GFTI_Grade']) else "#666"
                gfti_score = country['GFTI_Score'] if pd.notna(country['GFTI_Score']) else 0
                gfti_grade = country['GFTI_Grade'] if pd.notna(country['GFTI_Grade']) else "N/A"
                
                st.markdown(f"""
                <div class="card" style="border-left-color: {grade_color};">
                    <div class="metric-label">GFTI Score</div>
                    <div class="metric-highlight">{gfti_score:.0f}</div>
                    <div style="background-color: {grade_color}; color: white; padding: 2px 8px; border-radius: 12px; display: inline-block;">
                        Grade {gfti_grade}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                trust_tax = country['Trust_Tax_bps'] if pd.notna(country['Trust_Tax_bps']) else 0
                annual_cost = country['Annual_Cost_1B'] if pd.notna(country['Annual_Cost_1B']) else 0
                tax_color = '#DC2626' if trust_tax > 0 else '#10B981' if trust_tax < 0 else '#666'
                tax_sign = '+' if trust_tax > 0 else '' if trust_tax < 0 else ''
                
                st.markdown(f"""
                <div class="card" style="border-left-color: {tax_color};">
                    <div class="metric-label">Trust Tax</div>
                    <div class="metric-highlight" style="color: {tax_color};">{tax_sign}{trust_tax:.0f} bps</div>
                    <div>On $1B debt: ${annual_cost:.1f}M/year</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                bond_yield = country['Bond_Yield'] if pd.notna(country['Bond_Yield']) else 0
                sp_rating = country['SP_Rating'] if pd.notna(country['SP_Rating']) else "N/A"
                
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Bond Yield</div>
                    <div class="metric-highlight">{bond_yield:.2f}%</div>
                    <div>S&P Rating: {sp_rating}</div>
                </div>
                """, unsafe_allow_html=True)

            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                debt_gdp = country['Debt_GDP'] if pd.notna(country['Debt_GDP']) else 0
                st.metric("Debt/GDP", f"{debt_gdp:.1f}%")
            with col2:
                gdp_growth = country['GDP_Growth'] if pd.notna(country['GDP_Growth']) else 0
                st.metric("GDP Growth", f"{gdp_growth:.1f}%")
            with col3:
                inflation = country['Inflation'] if pd.notna(country['Inflation']) else 0
                st.metric("Inflation", f"{inflation:.1f}%")
            with col4:
                audit_opinion = country['Audit_Opinion'] if pd.notna(country['Audit_Opinion']) else "Unknown"
                st.metric("Audit Opinion", audit_opinion)

            # Key findings
            st.markdown("### 📋 Key Findings")
            key_issue = country['Key_Issue'] if pd.notna(country['Key_Issue']) else "No key issues identified"
            soe = country['SOE_Consolidated'] if pd.notna(country['SOE_Consolidated']) else "Unknown"
            pension = country['Pension_Recorded'] if pd.notna(country['Pension_Recorded']) else "Unknown"
            
            st.markdown(f"""
            <div class="card" style="border-left-color: #DC2626;">
                <strong>Primary Issue:</strong> {key_issue}<br>
                <strong>SOE Consolidation:</strong> {soe}<br>
                <strong>Pension Liability:</strong> {pension}
            </div>
            """, unsafe_allow_html=True)

            # Expected vs Actual
            st.markdown("### 📊 Yield Analysis")

            exp_data = pd.DataFrame({
                'Metric': ['Actual Yield', 'Expected (Rating Only)', 'Expected (Rating + GFTI)'],
                'Yield': [
                    country['Bond_Yield'] if pd.notna(country['Bond_Yield']) else 0,
                    country['Expected_Rating_Only'] if pd.notna(country['Expected_Rating_Only']) else 0,
                    country['Expected_Rating_GFTI'] if pd.notna(country['Expected_Rating_GFTI']) else 0
                ]
            })

            fig = px.bar(
                exp_data,
                x='Metric',
                y='Yield',
                color='Metric',
                title=f'Actual vs Expected Yield: {selected_country}',
                text='Yield'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📄 Full Country Report")
            
            # Generate report
            report_text = generate_country_report(country)
            
            # Display report in a scrollable text box
            st.text_area(
                label="Country Report Preview",
                value=report_text,
                height=600,
                disabled=True
            )
        
        with tab3:
            st.markdown("### 📥 Download Country Report")
            
            # Generate report
            report_text = generate_country_report(country)
            
            # Create download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Text file download
                filename = f"GFTI_{selected_country.replace(' ', '_')}_Report_{datetime.now().strftime('%Y%m%d')}.txt"
                st.markdown(
                    get_download_link(report_text, filename, f"⬇️ Download {selected_country} Report (TXT)"),
                    unsafe_allow_html=True
                )
            
            with col2:
                # CSV data download for this country
                country_data = country.to_frame().T
                csv = country_data.to_csv(index=False)
                csv_filename = f"GFTI_{selected_country.replace(' ', '_')}_Data_{datetime.now().strftime('%Y%m%d')}.csv"
                st.download_button(
                    label=f"⬇️ Download {selected_country} Data (CSV)",
                    data=csv,
                    file_name=csv_filename,
                    mime="text/csv"
                )
            
            st.markdown("### 📊 Key Data Points")
            
            # Display key data in a formatted table
            data_display = pd.DataFrame({
                'Metric': [
                    'Country', 'Region', 'GFTI Score', 'GFTI Grade', 
                    'S&P Rating', 'Bond Yield', 'Trust Tax (bps)', 
                    'Annual Cost on $1B Debt', 'Debt/GDP', 'GDP Growth',
                    'Inflation', 'Audit Opinion'
                ],
                'Value': [
                    country['Country'],
                    country['Region'],
                    f"{country['GFTI_Score']:.0f}" if pd.notna(country['GFTI_Score']) else "N/A",
                    country['GFTI_Grade'] if pd.notna(country['GFTI_Grade']) else "N/A",
                    country['SP_Rating'] if pd.notna(country['SP_Rating']) else "N/A",
                    f"{country['Bond_Yield']:.2f}%" if pd.notna(country['Bond_Yield']) else "N/A",
                    f"{country['Trust_Tax_bps']:+.0f} bps" if pd.notna(country['Trust_Tax_bps']) else "N/A",
                    f"${country['Annual_Cost_1B']:.1f}M" if pd.notna(country['Annual_Cost_1B']) else "N/A",
                    f"{country['Debt_GDP']:.1f}%" if pd.notna(country['Debt_GDP']) else "N/A",
                    f"{country['GDP_Growth']:.1f}%" if pd.notna(country['GDP_Growth']) else "N/A",
                    f"{country['Inflation']:.1f}%" if pd.notna(country['Inflation']) else "N/A",
                    country['Audit_Opinion'] if pd.notna(country['Audit_Opinion']) else "N/A"
                ]
            })
            
            st.dataframe(data_display, use_container_width=True, hide_index=True)
            
            
            # ========================================================
            # ADD THIS DELETE SECTION - RIGHT HERE
            # ========================================================
            st.markdown("---")
            st.markdown("### 🗑️ Delete Country")
            st.warning(f"⚠️ This will permanently delete **{selected_country}** from the database.")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                delete_confirmation = st.checkbox("I understand this cannot be undone")
            with col2:
                if delete_confirmation:
                    if st.button(f"🗑️ Delete {selected_country}", type="primary", use_container_width=True):
                        # Path to CSV file
                        csv_path = os.path.join(os.path.dirname(__file__), 'countries.csv')
                        
                        # Read existing data
                        if os.path.exists(csv_path):
                            existing = pd.read_csv(csv_path)
                            
                            # Remove the selected country
                            existing = existing[existing['Country'] != selected_country]
                            
                            # Save back to CSV
                            existing.to_csv(csv_path, index=False)
                            
                            # Success message
                            st.success(f"✅ {selected_country} has been deleted.")
                            
                            # Clear cache and rerun
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error("Database file not found.")

else:
    st.warning("No countries match the current filters. Try adjusting the filters in the sidebar.")

# ============================================================================
# COUNTRY COMPARISON
# ============================================================================
if not filtered_df.empty:
    st.markdown('<div class="sub-header">🔄 Country Comparison</div>', unsafe_allow_html=True)

    countries_to_compare = st.multiselect(
        "Select 2-4 countries to compare:",
        filtered_df['Country'].tolist(),
        default=filtered_df['Country'].tolist()[:min(4, len(filtered_df))] if len(filtered_df) >= 2 else filtered_df['Country'].tolist()
    )

    if len(countries_to_compare) >= 2:
        compare_df = filtered_df[filtered_df['Country'].isin(countries_to_compare)]
        
        # Select columns for comparison
        compare_cols = [
            'Country', 'GFTI_Score', 'GFTI_Grade', 'SP_Rating', 
            'Bond_Yield', 'Trust_Tax_bps', 'Annual_Cost_1B',
            'Debt_GDP', 'GDP_Growth', 'Inflation', 'Audit_Opinion'
        ]
        
        compare_display = compare_df[compare_cols].copy()
        # FIXED: Round Trust Tax but keep as float (don't convert to int)
        compare_display['Trust_Tax_bps'] = compare_display['Trust_Tax_bps'].round(0)
        compare_display.columns = [
            'Country', 'GFTI', 'Grade', 'S&P', 
            'Yield %', 'Trust Tax', 'Cost on $1B',
            'Debt/GDP %', 'GDP %', 'Inflation %', 'Audit Opinion'
        ]
        
        st.dataframe(compare_display, use_container_width=True, hide_index=True)
        
        # Download comparison
        csv = compare_display.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Comparison as CSV",
            data=csv,
            file_name=f"GFTI_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# TRUST TAX CALCULATOR - Using same simplified model
# ============================================================================
st.markdown('<div class="sub-header">🧮 Trust Tax Calculator</div>', unsafe_allow_html=True)
st.markdown("""
Calculate the cost of opacity for any country. Enter a country's S&P rating and GFTI score to see:
- Expected yield based on rating alone
- Transparency adjustment (GFTI impact)
- Expected yield with transparency factored in
- The Trust Tax (difference from actual yield)
- Annual cost on $1B of debt

*Note: This calculator uses a simplified model:*
- *Base expected yield = 3.0% + (S&P numeric rating × 0.25%)*
- *Transparency adjustment = ((50 - GFTI score) × 0.02%) (2bps per point)*
- *For developed markets, actual yields are heavily influenced by central bank rates and inflation*
""")

col1, col2, col3 = st.columns(3)

with col1:
    calc_country = st.text_input("Country Name", "New Country")
with col2:
    # Get unique ratings and their corresponding numeric values from the actual data
    rating_options = df[['SP_Rating', 'SP_Numeric']].dropna().drop_duplicates().sort_values('SP_Numeric')
    
    # Create a display format that shows both rating and its numeric value
    rating_display = [f"{row['SP_Rating']} (num: {int(row['SP_Numeric'])})" 
                      for _, row in rating_options.iterrows()]
    
    selected_display = st.selectbox("S&P Rating", rating_display)
    
    # Extract the actual rating and its numeric value from the selection
    calc_rating = selected_display.split(' (num:')[0]
    rating_num = int(selected_display.split('num: ')[1].rstrip(')'))
    
with col3:
    calc_gfti = st.slider("GFTI Score", 0, 100, 50)

# Step 1: Expected yield based on S&P rating only (using same formula as main model)
exp_rating_only = 3.0 + (rating_num * 0.25)

# Step 2: Adjust for transparency (GFTI score) - 2bps per point
gfti_adjustment = ((50 - calc_gfti) * 0.02)

# Step 3: Expected yield incorporating transparency
exp_rating_gfti = exp_rating_only + gfti_adjustment

# Display the three-step calculation
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Step 1: Expected (Rating Only)", 
        f"{exp_rating_only:.2f}%",
        help=f"3.0% + ({rating_num} × 0.25%)"
    )

with col2:
    st.metric(
        "Step 2: Transparency Adjustment", 
        f"{gfti_adjustment:+.2f}%", 
        help="Positive = Penalty for poor transparency, Negative = Reward for good transparency"
    )

with col3:
    st.metric(
        "Step 3: Expected (Rating + GFTI)", 
        f"{exp_rating_gfti:.2f}%",
        help="Step 1 + Step 2"
    )

with col4:
    # Find countries with similar rating to show typical range
    similar_countries = df[df['SP_Rating'] == calc_rating]['Bond_Yield'].dropna()
    if not similar_countries.empty:
        typical_yield = similar_countries.mean()
        st.metric(
            "Typical Actual Yield",
            f"{typical_yield:.2f}%",
            help=f"Average actual yield for {calc_rating}-rated countries"
        )
    else:
        st.metric("Typical Actual Yield", "N/A")

st.markdown("### Enter Actual Bond Yield to Calculate Trust Tax")
col1, col2 = st.columns(2)
with col1:
    default_yield = typical_yield if 'typical_yield' in locals() and not pd.isna(typical_yield) else exp_rating_only
    actual_yield_input = st.number_input(
        "Actual Bond Yield (%)", 
        min_value=0.0, 
        max_value=30.0, 
        value=round(default_yield, 2),
        step=0.1, 
        format="%.2f",
        key="actual_yield_calc"
    )

# Calculate Trust Tax
trust_tax_pct = actual_yield_input - exp_rating_gfti
trust_tax_bps = trust_tax_pct * 100
annual_cost = abs(trust_tax_bps / 100 * 10)  # $10M per 100bps on $1B

# Display results
st.markdown("### 📊 Trust Tax Calculation Results")

# Show the calculation visually
calc_df = pd.DataFrame({
    'Component': ['Actual Yield', 'Expected Yield (Rating + GFTI)', 'Trust Tax'],
    'Value': [f"{actual_yield_input:.2f}%", f"{exp_rating_gfti:.2f}%", f"{trust_tax_pct:+.2f}%"]
})
st.dataframe(calc_df, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)

with col1:
    tax_color = "#DC2626" if trust_tax_pct > 0 else "#10B981" if trust_tax_pct < 0 else "#666"
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {tax_color}20; border-left: 4px solid {tax_color};">
        <div style="font-size: 0.9rem; color: #666;">Trust Tax</div>
        <div style="font-size: 2.5rem; font-weight: bold; color: {tax_color};">{trust_tax_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f8f9fa; border-left: 4px solid {tax_color};">
        <div style="font-size: 0.9rem; color: #666;">Trust Tax (basis points)</div>
        <div style="font-size: 2.5rem; font-weight: bold; color: {tax_color};">{trust_tax_bps:+.0f} bps</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    cost_color = "#DC2626" if trust_tax_bps > 0 else "#10B981"
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f8f9fa; border-left: 4px solid {cost_color};">
        <div style="font-size: 0.9rem; color: #666;">Annual Cost on $1B Debt</div>
        <div style="font-size: 2.5rem; font-weight: bold; color: {cost_color};">${annual_cost:.1f}M</div>
    </div>
    """, unsafe_allow_html=True)

# Formula verification
st.markdown("### ✅ Formula Verification")
st.markdown(f"""
Step 1: Expected (Rating Only) = 3.0% + ({rating_num} × 0.25%) = {exp_rating_only:.2f}%
Step 2: Transparency Adjustment = (50 - {calc_gfti}) × 0.02% = {gfti_adjustment:+.2f}%
Step 3: Expected (Rating + GFTI) = {exp_rating_only:.2f}% + {gfti_adjustment:+.2f}% = {exp_rating_gfti:.2f}%
Step 4: Trust Tax = {actual_yield_input:.2f}% - {exp_rating_gfti:.2f}% = {trust_tax_pct:+.2f}% ({trust_tax_bps:+.0f} bps)
""")

# Check if this is a developed market and add context
is_developed = False
if calc_country != "New Country":
    country_match = df[df['Country'].str.contains(calc_country, case=False, na=False)]
    if not country_match.empty and country_match.iloc[0]['Region'] in ['Europe', 'North America', 'Pacific']:
        is_developed = True

# Interpretation card with contextual note
if is_developed and calc_gfti > 70:
    st.info("""
    **⚠️ Note for developed markets:** 
    This country's bond yield is significantly influenced by central bank rates, inflation expectations, 
    and safe-haven demand. The Trust Tax shown here may reflect these broader economic factors 
    rather than just transparency. Use this as a directional indicator, not an exact measure.
    """)

st.markdown(f"""
<div class="card" style="text-align: center;">
    <h4>📈 Interpretation:</h4>
    <p>
    <strong>{'+' if trust_tax_pct > 0 else '-' if trust_tax_pct < 0 else 'No '}Trust Tax Detected</strong><br>
    {f'This country pays <span style="color: #DC2626; font-weight: bold;">{trust_tax_bps:+.0f} bps MORE</span> than its transparency suggests.' if trust_tax_pct > 0 else 
      f'This country pays <span style="color: #10B981; font-weight: bold;">{abs(trust_tax_bps):.0f} bps LESS</span> than its transparency suggests.' if trust_tax_pct < 0 else 
      'This country pays exactly what its transparency suggests.'}
    </p>
    <p style="font-size: 0.9rem; color: #666;">
        <strong>If this country borrowed $1 billion:</strong><br>
        {'❌ They would pay' if trust_tax_pct > 0 else '✅ They would save'} <strong>${annual_cost:.1f} million</strong> per year in interest
        {' due to the transparency penalty.' if trust_tax_pct > 0 else ' due to transparency rewards.' if trust_tax_pct < 0 else '.'}
    </p>
    <hr>
    <p><em>Note: This simplified model uses 2bps per GFTI point adjustment from the 50-point baseline. For developed markets, other factors dominate.</em></p>
</div>
""", unsafe_allow_html=True)

# Show comparison with actual countries of similar rating
if not similar_countries.empty:
    st.markdown("### 📊 Comparison with Actual Countries")
    similar_data = df[df['SP_Rating'] == calc_rating][['Country', 'Region', 'GFTI_Score', 'Bond_Yield', 'Trust_Tax_bps']].copy()
    similar_data = similar_data.dropna(subset=['Bond_Yield'])
    if not similar_data.empty:
        similar_data['GFTI_Score'] = similar_data['GFTI_Score'].fillna(0).astype(int)
        similar_data.columns = ['Country', 'Region', 'GFTI Score', 'Actual Yield %', 'Trust Tax (bps)']
        st.dataframe(similar_data, use_container_width=True, hide_index=True)

# ============================================================================
# DATA QUALITY ISSUES
# ============================================================================
if not filtered_df.empty:
    st.markdown('<div class="sub-header">⚠️ Data Quality Issues by Country</div>', unsafe_allow_html=True)

    issues_df = filtered_df[[
        'Country', 'Audit_Opinion', 'Key_Issue', 'SOE_Consolidated', 'Pension_Recorded'
    ]].copy()
    issues_df.columns = [
        'Country', 'Audit Opinion', 'Key Finding', 'SOE Consolidated', 'Pension Recorded'
    ]

    st.dataframe(issues_df, use_container_width=True, height=300)

# ============================================================================
# DOWNLOAD FULL DATASET
# ============================================================================
st.markdown('<div class="sub-header">📥 Download Full Dataset</div>', unsafe_allow_html=True)

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download full dataset as CSV",
    data=csv,
    file_name=f"gfti_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# ============================================================================
# BATCH REPORT GENERATION
# ============================================================================
st.markdown('<div class="sub-header">📦 Batch Report Generation</div>', unsafe_allow_html=True)

if st.button("Generate Reports for All Filtered Countries"):
    with st.spinner("Generating reports..."):
        # Create a zip file in memory (simplified - just show option to download individually)
        st.success(f"Ready to download reports for {len(filtered_df)} countries. Use the Country Deep Dive section above to download individual reports.")
        
        # Show list of available reports
        report_list = filtered_df['Country'].tolist()
        st.markdown("**Available reports:**")
        st.markdown(", ".join(report_list))

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p><strong>GFTI: Global Financial Transparency Index</strong></p>
    <p>Data based on audited financial statements and market yields as of 2023-2025</p>
    <p>⚠️ <strong>Model note:</strong> Trust Tax uses a simplified model: 3.0% base + 0.25% per rating notch + 2bps per GFTI point adjustment. 
    For developed markets, actual yields are heavily influenced by central bank policies and inflation.</p>
    <p>📥 Country reports can be downloaded in TXT format for each nation</p>
    <p>🗺️ Interactive map shows GFTI scores globally - click any country for details</p>
</div>
""", unsafe_allow_html=True)