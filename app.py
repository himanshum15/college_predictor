

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from PIL import Image
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="College Predictor 2025",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to beautify the app
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .chance-high {
        background-color: #c8e6c9;
        padding: 5px 10px;
        border-radius: 10px;
        font-weight: bold;
        color: #2e7d32;
    }
    .chance-good {
        background-color: #bbdefb;
        padding: 5px 10px;
        border-radius: 10px;
        font-weight: bold;
        color: #1565c0;
    }
    .chance-tough {
        background-color: #ffecb3;
        padding: 5px 10px;
        border-radius: 10px;
        font-weight: bold;
        color: #ff8f00;
    }
    .chance-no {
        background-color: #ffcdd2;
        padding: 5px 10px;
        border-radius: 10px;
        font-weight: bold;
        color: #c62828;
    }
    .stDataFrame {
        font-size: 14px;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f0f7ff;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .sidebar-content {
        padding: 15px 10px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .download-btn {
        background-color: #1E88E5;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        margin-top: 10px;
    }
    .card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #eee;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the predicted cutoff data
@st.cache_data
def load_data():
    df = pd.read_csv('predicted_cutoffs_2025_2.csv')
    return df

# Chance categorization function with enhanced output
def categorize_chance(student_rank, predicted_closing_rank):
    if student_rank <= 0.7 * predicted_closing_rank:
        return "chance-high", "High Chance", "#c8e6c9"
    elif student_rank <= 0.9 * predicted_closing_rank:
        return "chance-good", "Good Chance", "#bbdefb"
    elif student_rank <= 1.05 * predicted_closing_rank:
        return "chance-tough", "Tough Chance", "#ffecb3"
    else:
        return "chance-no", "No Chance", "#ffcdd2"

# Create a better-looking histogram for rank distribution
def create_rank_distribution_chart(filtered_df):
    chart_data = filtered_df[['Academic Program Name', 'Predicted Closing Rank 2025']].sort_values('Predicted Closing Rank 2025')
    
    # Limit to top 15 colleges for better readability
    if len(chart_data) > 15:
        chart_data = chart_data.head(15)
    
    # Create a horizontal bar chart
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Predicted Closing Rank 2025:Q', title='Predicted Closing Rank 2025'),
        y=alt.Y('Academic Program Name:N', 
                sort=alt.EncodingSortField(field='Predicted Closing Rank 2025', order='ascending'),
                title='Academic Program'),
        color=alt.Color('Predicted Closing Rank 2025:Q', 
                       scale=alt.Scale(scheme='blues'),
                       legend=None),
        tooltip=['Academic Program Name', 'Predicted Closing Rank 2025']
    ).properties(
        title='Predicted Closing Ranks for 2025',
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        font='Arial',
        anchor='middle'
    )
    
    return chart

# Create trend line chart for historical closing ranks
def create_trend_chart(filtered_df):
    # Prepare data for trend chart - take top 5 colleges
    top_colleges = filtered_df.head(5)
    
    # Melt the dataframe to get it in the right format for Altair
    melted_df = pd.melt(
        top_colleges,
        id_vars=['Academic Program Name'],
        value_vars=['Closing Rank 2022', 'Closing Rank 2023', 'Closing Rank 2024', 'Predicted Closing Rank 2025'],
        var_name='Year',
        value_name='Closing Rank'
    )
    
    # Extract just the year from the Year column
    melted_df['Year'] = melted_df['Year'].str.extract('(\d{4})').astype(int)
    
    # Create the line chart
    chart = alt.Chart(melted_df).mark_line(point=True).encode(
        x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Closing Rank:Q', title='Closing Rank'),
        color=alt.Color('Academic Program Name:N', title='Program'),
        tooltip=['Academic Program Name', 'Year', 'Closing Rank']
    ).properties(
        title='Closing Rank Trends (2022-2025)',
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        font='Arial',
        anchor='middle'
    ).configure_legend(
        labelFontSize=10,
        titleFontSize=12
    )
    
    return chart

# Create a function to format the chance category
def format_chance(chance_class, chance_text):
    return f'<span class="{chance_class}">{chance_text}</span>'

# Function to create a summary card
def create_summary_card(student_rank, filtered_df):
    if filtered_df.empty:
        return
        
    high_chance_count = sum(filtered_df['Student Chance Class'] == 'chance-high')
    good_chance_count = sum(filtered_df['Student Chance Class'] == 'chance-good')
    tough_chance_count = sum(filtered_df['Student Chance Class'] == 'chance-tough')
    no_chance_count = sum(filtered_df['Student Chance Class'] == 'chance-no')
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### Summary for Rank {student_rank}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High Chance", high_chance_count)
    with col2:
        st.metric("Good Chance", good_chance_count)
    with col3:
        st.metric("Tough Chance", tough_chance_count)
    with col4:
        st.metric("No Chance", no_chance_count)
        
    top_recommendation = filtered_df.iloc[0] if not filtered_df.empty else None
    if top_recommendation is not None:
        st.markdown("#### Top Recommendation")
        st.markdown(f"""
        <div class="highlight">
            <strong>Institute:</strong> {top_recommendation['Institute']}<br>
            <strong>Program:</strong> {top_recommendation['Academic Program Name']}<br>
            <strong>Predicted Rank 2025:</strong> {top_recommendation['Predicted Closing Rank 2025']}<br>
            <strong>Your Chance:</strong> {format_chance(top_recommendation['Student Chance Class'], top_recommendation['Student Chance Text'])}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Streamlit App
def main():
    apply_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 College Predictor App - 2025</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
        Find the best college matches based on your rank and preferences
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure the 'predicted_cutoffs_2025_2.csv' file is in the same directory as this app.")
        return

    # Sidebar content
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.markdown("## 📝 Enter Your Details")
    
    student_rank = st.sidebar.number_input("Your Rank", min_value=1, max_value=500000, value=1000)
    
    # Get unique values for dropdowns
    quota_options = sorted(df['Quota'].unique())
    seat_type_options = sorted(df['Seat Type'].unique())
    gender_options = sorted(df['Gender'].unique())
    
    quota = st.sidebar.selectbox("Select Quota", quota_options)
    seat_type = st.sidebar.selectbox("Select Seat Type", seat_type_options)
    gender = st.sidebar.selectbox("Select Gender", gender_options)
    
    st.sidebar.markdown("## 🔍 Optional Filters")
    institute_filter = st.sidebar.text_input("Search by Institute")
    program_filter = st.sidebar.text_input("Search by Program")
    min_predicted_rank = st.sidebar.number_input("Maximum Predicted Closing Rank", min_value=0, value=0)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add app information in sidebar
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.markdown("## ℹ️ About This App")
    st.sidebar.markdown("""
    This app helps students predict their college admission chances based on:
    - Historical closing ranks from 2022-2024
    - Predicted closing ranks for 2025
    - Student's JEE/NEET rank and other criteria
    
    **Chance Categories:**
    - <span class="chance-high">High Chance</span>: Your rank is ≤ 70% of predicted closing rank
    - <span class="chance-good">Good Chance</span>: Your rank is ≤ 90% of predicted closing rank
    - <span class="chance-tough">Tough Chance</span>: Your rank is ≤ 105% of predicted closing rank
    - <span class="chance-no">No Chance</span>: Your rank is > 105% of predicted closing rank
    """, unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Filter data based on student input
    filtered_df = df[
        (df['Quota'] == quota) & 
        (df['Seat Type'] == seat_type) & 
        (df['Gender'] == gender)
    ].copy()

    # Apply optional filters
    if institute_filter:
        filtered_df = filtered_df[filtered_df['Institute'].str.contains(institute_filter, case=False, na=False)]
    if program_filter:
        filtered_df = filtered_df[filtered_df['Academic Program Name'].str.contains(program_filter, case=False, na=False)]
    if min_predicted_rank > 0:
        filtered_df = filtered_df[filtered_df['Predicted Closing Rank 2025'] <= min_predicted_rank]

    if filtered_df.empty:
        st.warning("⚠️ No matching programs found. Try changing your inputs.")
        return

    # Apply chance categorization and add colored labels
    filtered_df[['Student Chance Class', 'Student Chance Text', 'Student Chance Color']] = filtered_df.apply(
        lambda x: pd.Series(categorize_chance(student_rank, x['Predicted Closing Rank 2025'])),
        axis=1
    )
    
    # Define a custom sort order for chances
    chance_priority = {
        "chance-high": 1,
        "chance-good": 2,
        "chance-tough": 3,
        "chance-no": 4
    }
    filtered_df['Chance Priority'] = filtered_df['Student Chance Class'].map(chance_priority)

    # Sort by Chance Priority first, then by Predicted Closing Rank (lower is better)
    filtered_df = filtered_df.sort_values(
        by=['Chance Priority', 'Predicted Closing Rank 2025'],
        ascending=[True, True]
    )

    # Create summary card
    create_summary_card(student_rank, filtered_df)
    
    # Results section
    st.markdown('<h2 class="sub-header">🔍 Detailed Prediction Results</h2>', unsafe_allow_html=True)
    
    # Format the dataframe for display
    display_df = filtered_df[[
        'Institute',
        'Academic Program Name',
        'Closing Rank 2022',
        'Closing Rank 2023',
        'Closing Rank 2024',
        'Predicted Closing Rank 2025',
        'Student Chance Text',
        'Student Chance Color'  # Added color information
    ]].copy()
    
    # Rename the columns for better display
    display_df.columns = [
        'Institute',
        'Program',
        'Rank 2022',
        'Rank 2023',
        'Rank 2024',
        'Predicted 2025',
        'Your Chance',
        'Chance Color'  # Added color information
    ]
    
    # Create a function to style the entire dataframe with cell backgrounds
    def style_dataframe(df):
        # Create a copy to avoid modifying the original
        styled_df = df.copy()
        
        # Apply background colors to the "Your Chance" column
        styles = []
        for i in range(len(df)):
            # Create a style for each row
            row_styles = [''] * len(df.columns)
            
            # Set the background color for the "Your Chance" column (index 6)
            chance_color = df.iloc[i]['Chance Color']
            row_styles[6] = f'background-color: {chance_color}; color: black; font-weight: bold; text-align: center;'
            
            styles.append(row_styles)
        
        return styles
    
    def color_chance_column(val, color_map):
        if val.name == 'Your Chance':
            return [f'background-color: {color}; color: black; font-weight: bold; text-align: center;' 
                    for color in color_map]
        return [''] * len(color_map)
    
    # Apply styling to the dataframe
    styled_df = display_df.drop(columns=['Chance Color'])  # Remove the color column before display
    color_map = display_df['Chance Color'].tolist()

    st.dataframe(
        styled_df.style.apply(lambda s: color_chance_column(s, color_map), axis=0),
        use_container_width=True,
        height=400
    )


    # Download button
    csv = styled_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name='college_predictions_2025.csv',
        mime='text/csv',
        key='download-csv'
    )
    
    # Data visualization section
    st.markdown('<h2 class="sub-header">📊 Data Visualization</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.altair_chart(create_rank_distribution_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.altair_chart(create_trend_chart(filtered_df), use_container_width=True)
        
    # Tips section
    st.markdown('<h2 class="sub-header">💡 Tips for College Selection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <ol>
        <li><strong>Consider Programs with "High" or "Good" Chance</strong> - These are the programs where your rank gives you a strong chance of admission.</li>
        <li><strong>Look at Trend Data</strong> - If a program's closing rank has been steadily increasing over the years, it could indicate growing competition.</li>
        <li><strong>Don't Rely Solely on Rank</strong> - Also consider factors like location, infrastructure, faculty, and placement records.</li>
        <li><strong>Have Backup Options</strong> - Include some programs with "Tough Chance" in your choices as a backup plan.</li>
        <li><strong>Research Beyond Predictions</strong> - These are predictions based on historical data. Always research the latest information about colleges.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;">
        <p>College Predictor App - 2025 | Data based on historical JEE/NEET closing ranks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
