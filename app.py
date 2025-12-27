import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Synthetic Dataset Creation
def create_synthetic_dataset():
    # Districts of Andhra Pradesh
    districts = [
        'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna',
        'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 'Visakhapatnam',
        'Vizianagaram', 'West Godavari', 'YSR Kadapa'
    ]
    
    # Common crops in Andhra Pradesh (including new crops)
    crops = [
        'Rice', 'Maize', 'Corn', 'Cotton', 'Groundnut', 'Red Gram (Toor Dal)',
        'Green Gram (Moong Dal)', 'Black Gram (Urad Dal)', 'Sugarcane',
        'Chilli', 'Pepper', 'Turmeric', 'Tobacco', 'Sweet Potato', 'Mango', 
        'Banana', 'Coconut', 'Cashew', 'Soybean', 'Sunflower', 
        'Jowar (Sorghum)', 'Bajra (Pearl Millet)'
    ]
    
    # Months
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # Create synthetic data
    np.random.seed(42)
    num_samples = 5000
    
    data = {
        'District': np.random.choice(districts, num_samples),
        'Month': np.random.choice(months, num_samples),
        'Temperature': np.random.uniform(20, 40, num_samples),
        'Rainfall': np.random.uniform(0, 300, num_samples),
        'Soil_Type': np.random.choice(['Black', 'Red', 'Alluvial', 'Laterite'], num_samples),
        'Crop': np.random.choice(crops, num_samples),
        'Suitability': np.random.choice([0, 1], num_samples, p=[0.3, 0.7])
    }
    
    # Add some logical patterns based on real-world knowledge
    for i in range(num_samples):
        district = data['District'][i]
        month = data['Month'][i]
        
        # Adjust temperature based on month
        if month in ['December', 'January', 'February']:
            data['Temperature'][i] = np.random.uniform(15, 28)
        elif month in ['March', 'April', 'May']:
            data['Temperature'][i] = np.random.uniform(28, 42)
        else:
            data['Temperature'][i] = np.random.uniform(25, 35)
        
        # Adjust rainfall based on district and month
        if district in ['Visakhapatnam', 'Srikakulam', 'Vizianagaram']:
            if month in ['July', 'August', 'September']:
                data['Rainfall'][i] = np.random.uniform(150, 300)
            else:
                data['Rainfall'][i] = np.random.uniform(50, 150)
        elif district in ['Anantapur', 'Kurnool', 'YSR Kadapa']:
            data['Rainfall'][i] = np.random.uniform(0, 100)
        else:
            if month in ['July', 'August', 'September']:
                data['Rainfall'][i] = np.random.uniform(100, 250)
            else:
                data['Rainfall'][i] = np.random.uniform(20, 100)
        
        # Adjust suitability based on some logical conditions
        crop = data['Crop'][i]
        
        # Rice needs more water
        if crop == 'Rice' and data['Rainfall'][i] < 100:
            data['Suitability'][i] = 0
        
        # Corn needs moderate water and warm temperature
        if crop == 'Corn' and (data['Rainfall'][i] < 50 or data['Temperature'][i] < 20):
            data['Suitability'][i] = 0
        
        # Pepper needs warm, humid conditions
        if crop == 'Pepper' and (data['Temperature'][i] < 20 or data['Rainfall'][i] < 100):
            data['Suitability'][i] = 0
        
        # Sweet Potato grows well in warm conditions with moderate rainfall
        if crop == 'Sweet Potato' and (data['Temperature'][i] < 20 or data['Rainfall'][i] > 250):
            data['Suitability'][i] = 0
        
        # Groundnut grows well in Anantapur
        if crop == 'Groundnut' and district == 'Anantapur':
            data['Suitability'][i] = 1
        
        # Coconut grows well in coastal areas
        if crop == 'Coconut' and district in ['East Godavari', 'West Godavari', 'Visakhapatnam']:
            data['Suitability'][i] = 1
        
        # Chilli grows well in Guntur
        if crop == 'Chilli' and district == 'Guntur':
            data['Suitability'][i] = 1
        
        # Corn grows well in Krishna and Guntur
        if crop == 'Corn' and district in ['Krishna', 'Guntur', 'West Godavari']:
            data['Suitability'][i] = 1
        
        # Pepper grows well in coastal and hilly areas
        if crop == 'Pepper' and district in ['Visakhapatnam', 'Srikakulam', 'Vizianagaram']:
            data['Suitability'][i] = 1
        
        # Sweet Potato grows well in various districts
        if crop == 'Sweet Potato' and district in ['East Godavari', 'West Godavari', 'Krishna']:
            data['Suitability'][i] = 1
    
    df = pd.DataFrame(data)
    return df, crops, districts, months

# Create dataset
df, crops, districts, months = create_synthetic_dataset()

# Train machine learning model
def train_model(df):
    # Convert categorical variables to numerical
    df_encoded = pd.get_dummies(df, columns=['District', 'Month', 'Soil_Type', 'Crop'])
    
    X = df_encoded.drop('Suitability', axis=1)
    y = df_encoded['Suitability']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

model = train_model(df)

# Crop information and precautions (updated with new crops)
crop_info = {
    'Rice': {
        'description': 'Staple food crop requiring abundant water',
        'precautions': [
            'Ensure proper water management (5-10 cm standing water)',
            'Use certified seeds for better yield',
            'Control weeds in early stages',
            'Monitor for pests like stem borers and leaf folders'
        ]
    },
    'Maize': {
        'description': 'Versatile cereal crop grown in diverse conditions',
        'precautions': [
            'Plant in well-drained soil',
            'Maintain proper spacing (60x20 cm)',
            'Apply nitrogen in split doses',
            'Watch for fall armyworm infestation'
        ]
    },
    'Corn': {
        'description': 'Sweet corn variety popular for direct consumption',
        'precautions': [
            'Plant in well-drained loamy soil',
            'Maintain spacing of 60x25 cm',
            'Harvest when silks turn brown and dry',
            'Control corn earworm and aphids',
            'Irrigate regularly during grain filling stage'
        ]
    },
    'Cotton': {
        'description': 'Important cash crop known as "white gold"',
        'precautions': [
            'Use Bt cotton seeds for pest resistance',
            'Monitor for pink bollworm',
            'Practice crop rotation to prevent soil depletion',
            'Avoid waterlogging in fields'
        ]
    },
    'Groundnut': {
        'description': 'Oilseed crop important for protein and oil',
        'precautions': [
            'Plant in well-drained sandy loam soil',
            'Apply gypsum at flowering stage',
            'Harvest at proper maturity to avoid aflatoxin',
            'Store in dry conditions'
        ]
    },
    'Red Gram (Toor Dal)': {
        'description': 'Important pulse crop rich in protein',
        'precautions': [
            'Drought resistant but needs irrigation at flowering',
            'Treat seeds with rhizobium culture',
            'Control pod borer with recommended pesticides',
            'Harvest when 80% pods are mature'
        ]
    },
    'Green Gram (Moong Dal)': {
        'description': 'Short duration pulse crop',
        'precautions': [
            'Grows well in well-drained soils',
            'Short duration (60-70 days)',
            'Susceptible to yellow mosaic virus - use resistant varieties',
            'Harvest when 80% pods are mature'
        ]
    },
    'Black Gram (Urad Dal)': {
        'description': 'Important pulse crop for protein',
        'precautions': [
            'Grows well in black cotton soils',
            'Treat seeds with rhizobium culture',
            'Control leaf spot diseases',
            'Harvest when pods turn black'
        ]
    },
    'Sugarcane': {
        'description': 'Important cash crop for sugar production',
        'precautions': [
            'Requires heavy irrigation',
            'Use disease-free setts for planting',
            'Control early shoot borer',
            'Harvest at proper maturity (10-12 months)'
        ]
    },
    'Chilli': {
        'description': 'Important spice crop with high value',
        'precautions': [
            'Requires well-drained fertile soil',
            'Irrigate carefully to avoid flower drop',
            'Control fruit borer and mites',
            'Harvest at color break stage'
        ]
    },
    'Pepper': {
        'description': 'Black pepper, important spice crop',
        'precautions': [
            'Plant in well-drained red loamy soil',
            'Provide support with standards or trellis',
            'Control quick wilt and pollu beetle',
            'Harvest when berries turn orange-red',
            'Provide shade during initial growth'
        ]
    },
    'Turmeric': {
        'description': 'Important spice crop with medicinal value',
        'precautions': [
            'Plant in well-drained fertile soil',
            'Treat seed rhizomes with fungicide',
            'Control leaf spot diseases',
            'Harvest after 8-9 months when leaves dry'
        ]
    },
    'Tobacco': {
        'description': 'Commercial crop mainly for export',
        'precautions': [
            'Requires well-drained sandy loam soils',
            'Needs careful curing after harvest',
            'Follow government regulations',
            'Practice crop rotation'
        ]
    },
    'Sweet Potato': {
        'description': 'Nutritious root vegetable rich in vitamins',
        'precautions': [
            'Plant in well-drained sandy loam soil',
            'Use vine cuttings for propagation',
            'Control sweet potato weevil',
            'Harvest when leaves turn yellow',
            'Cure properly before storage'
        ]
    },
    'Mango': {
        'description': 'Important fruit crop of Andhra Pradesh',
        'precautions': [
            'Plant in well-drained deep soils',
            'Prune for proper canopy management',
            'Control mango hopper and fruit fly',
            'Harvest at proper maturity'
        ]
    },
    'Banana': {
        'description': 'Important fruit crop with high yield',
        'precautions': [
            'Requires heavy irrigation and fertilization',
            'Plant disease-free tissue culture plants',
            'Control sigatoka leaf spot disease',
            'Support plants during fruiting'
        ]
    },
    'Coconut': {
        'description': 'Important plantation crop of coastal areas',
        'precautions': [
            'Plant in coastal sandy soils',
            'Apply balanced fertilizers regularly',
            'Control rhinoceros beetle',
            'Intercrop with cocoa or pepper'
        ]
    },
    'Cashew': {
        'description': 'Important plantation crop for export',
        'precautions': [
            'Plant in well-drained sandy soils',
            'Prune for proper shape',
            'Control tea mosquito bug',
            'Harvest nuts when apple turns pink'
        ]
    },
    'Soybean': {
        'description': 'Oilseed crop rich in protein',
        'precautions': [
            'Plant in well-drained soils',
            'Inoculate seeds with rhizobium',
            'Control yellow mosaic virus',
            'Harvest when leaves yellow and drop'
        ]
    },
    'Sunflower': {
        'description': 'Important oilseed crop',
        'precautions': [
            'Plant in well-drained soils',
            'Provide support if needed',
            'Control head borer',
            'Harvest when back of head turns yellow'
        ]
    },
    'Jowar (Sorghum)': {
        'description': 'Traditional millet crop',
        'precautions': [
            'Drought resistant crop',
            'Control shoot fly in early stages',
            'Harvest when grains are hard'
        ]
    },
    'Bajra (Pearl Millet)': {
        'description': 'Traditional drought-resistant crop',
        'precautions': [
            'Grows well in poor soils',
            'Control downy mildew',
            'Harvest when grains are hard'
        ]
    }
}

# District-wise climate information (unchanged)
district_climate = {
    'Anantapur': {
        'description': 'Hot and dry climate with low rainfall',
        'soil': 'Red sandy loam soils',
        'avg_temp': '28-40°C',
        'avg_rainfall': '500-600 mm'
    },
    'Chittoor': {
        'description': 'Moderate climate with some hilly areas',
        'soil': 'Red soils and black cotton soils',
        'avg_temp': '22-38°C',
        'avg_rainfall': '900-1000 mm'
    },
    'East Godavari': {
        'description': 'Coastal district with high humidity',
        'soil': 'Alluvial and deltaic soils',
        'avg_temp': '24-36°C',
        'avg_rainfall': '1000-1100 mm'
    },
    'Guntur': {
        'description': 'Coastal plains with hot climate',
        'soil': 'Black cotton soils',
        'avg_temp': '25-38°C',
        'avg_rainfall': '800-900 mm'
    },
    'Krishna': {
        'description': 'Coastal district with fertile delta',
        'soil': 'Alluvial and black soils',
        'avg_temp': '24-36°C',
        'avg_rainfall': '900-1000 mm'
    },
    'Kurnool': {
        'description': 'Semi-arid climate with low rainfall',
        'soil': 'Red soils and black soils',
        'avg_temp': '26-40°C',
        'avg_rainfall': '600-700 mm'
    },
    'Nellore': {
        'description': 'Coastal district with moderate rainfall',
        'soil': 'Red soils and sandy loams',
        'avg_temp': '24-36°C',
        'avg_rainfall': '1000-1100 mm'
    },
    'Prakasam': {
        'description': 'Mixed coastal and dry climate',
        'soil': 'Red soils and sandy loams',
        'avg_temp': '25-38°C',
        'avg_rainfall': '800-900 mm'
    },
    'Srikakulam': {
        'description': 'Northern coastal district with good rainfall',
        'soil': 'Red and alluvial soils',
        'avg_temp': '22-34°C',
        'avg_rainfall': '1100-1200 mm'
    },
    'Visakhapatnam': {
        'description': 'Coastal district with hilly terrain',
        'soil': 'Red and laterite soils',
        'avg_temp': '22-33°C',
        'avg_rainfall': '1000-1100 mm'
    },
    'Vizianagaram': {
        'description': 'Coastal district with moderate climate',
        'soil': 'Red and alluvial soils',
        'avg_temp': '23-35°C',
        'avg_rainfall': '1000-1100 mm'
    },
    'West Godavari': {
        'description': 'Fertile delta region with high humidity',
        'soil': 'Alluvial and black soils',
        'avg_temp': '24-36°C',
        'avg_rainfall': '1000-1100 mm'
    },
    'YSR Kadapa': {
        'description': 'Hot and dry climate with low rainfall',
        'soil': 'Red soils and black soils',
        'avg_temp': '27-40°C',
        'avg_rainfall': '600-700 mm'
    }
}

# Prediction function (unchanged)
def predict_crop(district, month, crop_choice=None):
    # Get current temperature and rainfall based on district and month
    temp = df[(df['District'] == district) & (df['Month'] == month)]['Temperature'].mean()
    rainfall = df[(df['District'] == district) & (df['Month'] == month)]['Rainfall'].mean()
    soil_type = df[df['District'] == district]['Soil_Type'].mode()[0]
    
    # Prepare input for model
    input_data = {
        'District': district,
        'Month': month,
        'Temperature': temp,
        'Rainfall': rainfall,
        'Soil_Type': soil_type
    }
    
    # If user has selected a crop
    if crop_choice and crop_choice != "I don't know":
        input_data['Crop'] = crop_choice
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df, columns=['District', 'Month', 'Soil_Type', 'Crop'])
        
        # Ensure all columns are present (add missing with 0)
        train_columns = pd.get_dummies(df, columns=['District', 'Month', 'Soil_Type', 'Crop']).columns.drop('Suitability')
        for col in train_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[train_columns]
        
        prediction = model.predict(input_encoded)[0]
        
        if prediction == 1:
            result = f"✅ {crop_choice} is suitable to grow in {district} during {month}."
            precautions = crop_info[crop_choice]['precautions']
            precautions_text = "\n".join([f"• {precaution}" for precaution in precautions])
            output = f"{result}\n\n📌 Precautions:\n{precautions_text}"
        else:
            alternatives = get_alternative_crops(district, month)
            alt_text = "\n".join([f"• {crop}" for crop in alternatives[:3]])
            output = f"❌ {crop_choice} is not recommended for {district} in {month}.\n\n🌱 Better alternatives:\n{alt_text}"
    else:
        # Recommend best crops
        recommended_crops = get_alternative_crops(district, month)
        rec_text = "\n".join([f"• {crop}" for crop in recommended_crops[:5]])
        
        # Get climate info
        climate = district_climate[district]
        climate_text = (
            f"🌡️ Avg Temperature: {climate['avg_temp']}\n"
            f"🌧️ Avg Rainfall: {climate['avg_rainfall']}\n"
            f"🌱 Soil Type: {climate['soil']}"
        )
        
        output = (
            f"🌾 Recommended crops for {district} in {month}:\n\n{rec_text}\n\n"
            f"📌 District Climate Info:\n{climate_text}"
        )
    
    return output

def get_alternative_crops(district, month):
    # Get current temperature and rainfall based on district and month
    temp = df[(df['District'] == district) & (df['Month'] == month)]['Temperature'].mean()
    rainfall = df[(df['District'] == district) & (df['Month'] == month)]['Rainfall'].mean()
    soil_type = df[df['District'] == district]['Soil_Type'].mode()[0]
    
    # Test all crops and get suitability scores
    crop_scores = []
    for crop in crops:
        input_data = {
            'District': district,
            'Month': month,
            'Temperature': temp,
            'Rainfall': rainfall,
            'Soil_Type': soil_type,
            'Crop': crop
        }
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df, columns=['District', 'Month', 'Soil_Type', 'Crop'])
        
        # Ensure all columns are present (add missing with 0)
        train_columns = pd.get_dummies(df, columns=['District', 'Month', 'Soil_Type', 'Crop']).columns.drop('Suitability')
        for col in train_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[train_columns]
        
        # Get probability instead of binary prediction
        proba = model.predict_proba(input_encoded)[0][1]
        crop_scores.append((crop, proba))
    
    # Sort by probability
    crop_scores.sort(key=lambda x: x[1], reverse=True)
    return [crop for crop, score in crop_scores if score > 0.7]

# Custom CSS for styling (unchanged)
css = """
.gradio-container {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.title {
    text-align: center;
    color: #2c3e50;
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.description {
    text-align: center;
    color: #4a5568;
    margin-bottom: 30px;
    font-size: 16px;
}
.input-section {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
.input-label {
    font-weight: 500;
    color: #2d3748;
    margin-bottom: 8px;
    display: block;
}
.output-section {
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    min-height: 200px;
    font-size: 16px;
    line-height: 1.6;
    white-space: pre-wrap;
}
.output-title {
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 15px;
    font-size: 20px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
}
.btn-primary {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    border: none;
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1);
}
.select-dropdown, .text-input {
    width: 100%;
    padding: 12px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-size: 16px;
    transition: all 0.3s ease;
}
.select-dropdown:focus, .text-input:focus {
    border-color: #4b6cb7;
    box-shadow: 0 0 0 3px rgba(75, 108, 183, 0.2);
    outline: none;
}
.footer {
    text-align: center;
    margin-top: 30px;
    color: #718096;
    font-size: 14px;
}
.success {
    color: #2e7d32;
}
.warning {
    color: #d32f2f;
}
.recommendation {
    background: #f0f4f8;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
    border-left: 4px solid #4b6cb7;
}
.crop-image {
    max-width: 100%;
    border-radius: 8px;
    margin-top: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""

# Gradio Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("""
    <div class="title">🌱 Applications of Machine-Learning  District-wise Climate,Soil & Water Based Crop Detection for Sustainable Cropping</div>
    
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Group(visible=True) as input_section:
                gr.Markdown("### 📍 Enter Your Farming Details")
                district = gr.Dropdown(
                    label="Select Your District",
                    choices=districts,
                    value="Krishna",
                    interactive=True,
                    elem_classes="select-dropdown"
                )
                month = gr.Dropdown(
                    label="Select Planting Month",
                    choices=months,
                    value=datetime.now().strftime("%B"),
                    interactive=True,
                    elem_classes="select-dropdown"
                )
                crop_choice = gr.Dropdown(
                    label="Do you have a specific crop in mind? (Select 'I don't know' for recommendations)",
                    choices=["I don't know"] + sorted(crops),
                    value="I don't know",
                    interactive=True,
                    elem_classes="select-dropdown"
                )
                submit_btn = gr.Button("Get Recommendation", variant="primary", elem_classes="btn-primary")
        
        with gr.Column():
            with gr.Group(visible=True) as output_section:
                # Remove the title from the output section
                output = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=15,
                    elem_classes="output-section"
                )
    
    gr.Markdown("""
    <div class="footer">
        Note: This system provides recommendations based on historical data and machine learning predictions. 
        Always consult with local agricultural experts before making final decisions.
    </div>
    """)
    
    submit_btn.click(
        fn=predict_crop,
        inputs=[district, month, crop_choice],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()