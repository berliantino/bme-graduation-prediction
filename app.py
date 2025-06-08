import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        text-align: center;
        margin: 2rem 0;
    }
    .upload-area {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Define directories
model_dir = "model"

# Load model function
@st.cache_resource
def load_model():
    model_path = os.path.join(model_dir, "student_status_pipeline_labeled.joblib")
    info_path = os.path.join(model_dir, "columns_info_labeled.joblib")
    
    try:
        model_pipeline = joblib.load(model_path)
        columns_info = joblib.load(info_path)
        return model_pipeline, columns_info, True
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure model files exist in the 'model' directory.")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, False

# Feature label mappings for better UI
FEATURE_LABELS = {
    'Marital_status': 'Status Pernikahan',
    'Application_mode': 'Mode Aplikasi',
    'Application_order': 'Urutan Aplikasi (0-9)',
    'Course': 'Program Studi',
    'Daytime_evening_attendance': 'Waktu Kuliah',
    'Previous_qualification': 'Kualifikasi Sebelumnya',
    'Previous_qualification_grade': 'Nilai Kualifikasi Sebelumnya',
    'Nacionality': 'Kewarganegaraan',
    'Mothers_qualification': 'Pendidikan Ibu',
    'Fathers_qualification': 'Pendidikan Ayah',
    'Mothers_occupation': 'Pekerjaan Ibu',
    'Fathers_occupation': 'Pekerjaan Ayah',
    'Admission_grade': 'Nilai Masuk',
    'Displaced': 'Status Pengungsi',
    'Educational_special_needs': 'Kebutuhan Khusus',
    'Debtor': 'Status Hutang',
    'Tuition_fees_up_to_date': 'SPP Terkini',
    'Gender': 'Jenis Kelamin',
    'Scholarship_holder': 'Penerima Beasiswa',
    'Age_at_enrollment': 'Usia Saat Mendaftar',
    'International': 'Mahasiswa Internasional',
    'Curricular_units_1st_sem_credited': 'SKS Diakui Semester 1',
    'Curricular_units_1st_sem_enrolled': 'SKS Diambil Semester 1',
    'Curricular_units_1st_sem_evaluations': 'Evaluasi Semester 1',
    'Curricular_units_1st_sem_approved': 'SKS Lulus Semester 1',
    'Curricular_units_1st_sem_grade': 'Nilai Semester 1',
    'Curricular_units_1st_sem_without_evaluations': 'Tanpa Evaluasi Semester 1',
    'Curricular_units_2nd_sem_credited': 'SKS Diakui Semester 2',
    'Curricular_units_2nd_sem_enrolled': 'SKS Diambil Semester 2',
    'Curricular_units_2nd_sem_evaluations': 'Evaluasi Semester 2',
    'Curricular_units_2nd_sem_approved': 'SKS Lulus Semester 2',
    'Curricular_units_2nd_sem_grade': 'Nilai Semester 2',
    'Curricular_units_2nd_sem_without_evaluations': 'Tanpa Evaluasi Semester 2',
    'Unemployment_rate': 'Tingkat Pengangguran (%)',
    'Inflation_rate': 'Tingkat Inflasi (%)',
    'GDP': 'GDP'
}

def create_prediction_visualization(prediction_proba, target_classes):
    """Create beautiful prediction visualization"""
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Probabilitas Prediksi", "Confidence Level"),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Bar chart for probabilities
    colors = ['#ff7f7f' if cls == 'Dropout' else '#90EE90' if cls == 'Graduate' else '#87CEEB' 
              for cls in target_classes]
    
    fig.add_trace(
        go.Bar(
            x=target_classes,
            y=prediction_proba[0] * 100,
            marker_color=colors,
            text=[f"{prob:.1f}%" for prob in prediction_proba[0] * 100],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Confidence indicator
    max_prob = max(prediction_proba[0]) * 100
    confidence_color = "green" if max_prob > 70 else "orange" if max_prob > 50 else "red"
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=max_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': confidence_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Hasil Prediksi Detail",
        title_x=0.5
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Prediksi Status Kelulusan Siswa</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Jaya Jaya Institut - Student Performance Prediction System</p>', unsafe_allow_html=True)
    
    # Load model
    model_pipeline, columns_info, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Extract model info
    numerical_features = columns_info["numerical"]
    categorical_features = columns_info["categorical"] 
    category_levels = columns_info["category_levels"]
    feature_order = columns_info["feature_order"]
    target_classes = columns_info["target_classes"]
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    mode = st.sidebar.radio(
        "Pilih Mode Prediksi:",
        ["📝 Input Manual", "📁 Upload CSV"],
        index=0
    )
    
    if mode == "📝 Input Manual":
        manual_prediction(model_pipeline, numerical_features, categorical_features, 
                         category_levels, feature_order, target_classes)
    else:
        csv_prediction(model_pipeline, numerical_features, categorical_features,
                      category_levels, feature_order, target_classes)

def manual_prediction(model_pipeline, numerical_features, categorical_features, 
                     category_levels, feature_order, target_classes):
    """Manual input prediction interface"""
    
    st.markdown('<h2 class="sub-header">📝 Input Data Siswa</h2>', unsafe_allow_html=True)
    
    # Organize features into categories for better UX
    basic_info = ['Marital_status', 'Gender', 'Age_at_enrollment', 'Nacionality', 'International']
    academic_info = ['Course', 'Daytime_evening_attendance', 'Previous_qualification', 
                    'Previous_qualification_grade', 'Admission_grade']
    family_info = ['Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation']
    financial_info = ['Debtor', 'Tuition_fees_up_to_date', 'Scholarship_holder']
    performance_info = [f for f in feature_order if 'Curricular_units' in f]
    application_info = ['Application_mode', 'Application_order']
    other_info = ['Displaced', 'Educational_special_needs', 'Unemployment_rate', 'Inflation_rate', 'GDP']
    
    # Create tabs for different categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👤 Info Dasar", "🎓 Akademik", "👨‍👩‍👧‍👦 Keluarga", "💰 Keuangan", 
        "📊 Performa", "📋 Aplikasi", "🔧 Lainnya"
    ])
    
    input_data = {}
    
    with tab1:
        st.markdown("### Informasi Dasar Siswa")
        cols = st.columns(2)
        for i, feature in enumerate(basic_info):
            if feature in feature_order:
                with cols[i % 2]:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    with tab2:
        st.markdown("### Informasi Akademik")
        cols = st.columns(2)
        for i, feature in enumerate(academic_info):
            if feature in feature_order:
                with cols[i % 2]:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    with tab3:
        st.markdown("### Informasi Keluarga")
        cols = st.columns(2)
        for i, feature in enumerate(family_info):
            if feature in feature_order:
                with cols[i % 2]:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    with tab4:
        st.markdown("### Informasi Keuangan")
        cols = st.columns(3)
        for i, feature in enumerate(financial_info):
            if feature in feature_order:
                with cols[i % 3]:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    with tab5:
        st.markdown("### Performa Akademik")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Semester 1**")
            sem1_features = [f for f in performance_info if '1st_sem' in f]
            for feature in sem1_features:
                if feature in feature_order:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
        
        with col2:
            st.markdown("**Semester 2**")
            sem2_features = [f for f in performance_info if '2nd_sem' in f]
            for feature in sem2_features:
                if feature in feature_order:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    with tab6:
        st.markdown("### Informasi Aplikasi")
        cols = st.columns(2)
        for i, feature in enumerate(application_info):
            if feature in feature_order:
                with cols[i % 2]:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    with tab7:
        st.markdown("### Informasi Lainnya")
        cols = st.columns(2)
        for i, feature in enumerate(other_info):
            if feature in feature_order:
                with cols[i % 2]:
                    input_data[feature] = create_input_field(feature, numerical_features, 
                                                           categorical_features, category_levels)
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Prediksi Status Siswa", type="primary", use_container_width=True):
            make_prediction(input_data, model_pipeline, numerical_features, 
                          categorical_features, category_levels, feature_order, target_classes)

def csv_prediction(model_pipeline, numerical_features, categorical_features,
                  category_levels, feature_order, target_classes):
    """CSV upload prediction interface"""
    
    st.markdown('<h2 class="sub-header">📁 Upload File CSV</h2>', unsafe_allow_html=True)
    
    # Sample CSV download
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown("### 📥 Download Template")
        
        # Create sample DataFrame for download
        try:
            sample_data = create_sample_data(feature_order, numerical_features, categorical_features, category_levels)
            csv_sample = sample_data.to_csv(index=False)
            
            st.download_button(
                label="📋 Download Template CSV",
                data=csv_sample,
                file_name="student_data_template.csv",
                mime="text/csv",
                help="Download template CSV dengan format yang benar"
            )
        except Exception as e:
            st.error(f"Error creating sample data: {e}")
            st.info("Template akan dibuat dengan struktur basic")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload File")
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=['csv'],
            help="Upload file CSV dengan kolom sesuai template"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diupload! {len(df)} baris data ditemukan.")
            
            # Show data preview
            with st.expander("👁️ Preview Data", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Validate columns
            missing_cols = set(feature_order) - set(df.columns)
            if missing_cols:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
                return
            
            # Prediction button for CSV
            if st.button("🔮 Prediksi Semua Data", type="primary", use_container_width=True):
                make_batch_prediction(df, model_pipeline, numerical_features, 
                                    categorical_features, category_levels, feature_order, target_classes)
                
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

def create_input_field(feature, numerical_features, categorical_features, category_levels):
    """Create appropriate input field for each feature"""
    
    label = FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())
    
    if feature in numerical_features:
        # Set appropriate ranges for different types of numerical features
        min_val, max_val, default_val = get_numerical_ranges(feature)
        
        return st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            key=f"input_{feature}",
            help=f"Masukkan nilai untuk {label}"
        )
    
    elif feature in categorical_features:
        options = category_levels.get(feature, [])
        if options:
            return st.selectbox(
                label,
                options=options,
                key=f"input_{feature}",
                help=f"Pilih {label}"
            )
        else:
            st.warning(f"⚠️ Opsi untuk {feature} tidak ditemukan")
            return None

def get_numerical_ranges(feature):
    """Get appropriate ranges for numerical features"""
    
    if "grade" in feature.lower():
        return 0.0, 200.0, 100.0
    elif "age" in feature.lower():
        return 17.0, 70.0, 20.0
    elif "curricular" in feature.lower() and ("unit" in feature.lower() or "evaluation" in feature.lower()):
        return 0.0, 30.0, 5.0
    elif "rate" in feature.lower():
        return -10.0, 20.0, 0.0
    elif "gdp" in feature.lower():
        return -5.0, 10.0, 2.0
    elif "order" in feature.lower():
        return 0.0, 9.0, 0.0
    else:
        return 0.0, 1000.0, 0.0

def create_sample_data(feature_order, numerical_features, categorical_features, category_levels):
    """Create sample data for CSV template - FIXED VERSION"""
    
    # Fixed number of rows for consistent array lengths
    num_rows = 3
    sample_data = {}
    
    for feature in feature_order:
        if feature in numerical_features:
            min_val, max_val, default_val = get_numerical_ranges(feature)
            # Create exactly num_rows values
            sample_data[feature] = [default_val, default_val + 10, default_val - 5][:num_rows]
            
        elif feature in categorical_features:
            options = category_levels.get(feature, ['Option1', 'Option2', 'Option3'])
            # Ensure we have exactly num_rows values
            if len(options) >= num_rows:
                sample_data[feature] = options[:num_rows]
            else:
                # Repeat options to get num_rows values
                repeated_options = (options * ((num_rows // len(options)) + 1))[:num_rows]
                sample_data[feature] = repeated_options
        else:
            # Fallback for unknown feature types
            sample_data[feature] = ['Unknown'] * num_rows
    
    # Verify all arrays have the same length
    lengths = [len(values) for values in sample_data.values()]
    if len(set(lengths)) > 1:
        st.error(f"Error: Inconsistent array lengths: {dict(zip(sample_data.keys(), lengths))}")
        # Fix by truncating all to minimum length
        min_length = min(lengths)
        for key in sample_data:
            sample_data[key] = sample_data[key][:min_length]
    
    return pd.DataFrame(sample_data)

def make_prediction(input_data, model_pipeline, numerical_features, categorical_features,
                   category_levels, feature_order, target_classes):
    """Make prediction for single input"""
    
    # Validate input
    if not all(val is not None for val in input_data.values()) or len(input_data) != len(feature_order):
        st.error("❌ Harap isi semua field yang diperlukan!")
        return
    
    try:
        # Prepare data
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order]
        
        # Convert data types
        for col in numerical_features:
            input_df[col] = pd.to_numeric(input_df[col])
        for col in categorical_features:
            if col in category_levels:
                input_df[col] = pd.Categorical(input_df[col], categories=category_levels[col])
        
        # Make prediction
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)
        
        # Display results
        st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
        
        # Main prediction result
        status_emoji = "🎓" if prediction[0] == "Graduate" else "❌" if prediction[0] == "Dropout" else "📚"
        st.markdown(f"### {status_emoji} Status Diprediksi: **{prediction[0]}**")
        
        # Confidence score
        confidence = max(prediction_proba[0]) * 100
        st.markdown(f"**Confidence Score: {confidence:.1f}%**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed visualization
        fig = create_prediction_visualization(prediction_proba, target_classes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show input data
        with st.expander("📋 Data Input yang Digunakan"):
            st.dataframe(input_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error dalam prediksi: {e}")

def make_batch_prediction(df, model_pipeline, numerical_features, categorical_features,
                         category_levels, feature_order, target_classes):
    """Make batch prediction for CSV data"""
    
    try:
        # Prepare data
        df_pred = df[feature_order].copy()
        
        # Convert data types
        for col in numerical_features:
            df_pred[col] = pd.to_numeric(df_pred[col])
        for col in categorical_features:
            if col in category_levels:
                df_pred[col] = pd.Categorical(df_pred[col], categories=category_levels[col])
        
        # Make predictions
        predictions = model_pipeline.predict(df_pred)
        prediction_proba = model_pipeline.predict_proba(df_pred)
        
        # Add results to dataframe
        df_results = df.copy()
        df_results['Predicted_Status'] = predictions
        
        for i, cls in enumerate(target_classes):
            df_results[f'Probability_{cls}'] = prediction_proba[:, i]
        
        # Display results
        st.success(f"✅ Prediksi selesai untuk {len(df)} data!")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            graduate_count = sum(predictions == 'Graduate')
            st.metric("🎓 Graduate", graduate_count, f"{graduate_count/len(predictions)*100:.1f}%")
        
        with col2:
            dropout_count = sum(predictions == 'Dropout')
            st.metric("❌ Dropout", dropout_count, f"{dropout_count/len(predictions)*100:.1f}%")
        
        with col3:
            enrolled_count = sum(predictions == 'Enrolled')
            st.metric("📚 Enrolled", enrolled_count, f"{enrolled_count/len(predictions)*100:.1f}%")
        
        # Results table
        st.markdown("### 📊 Hasil Prediksi Detail")
        st.dataframe(df_results, use_container_width=True)
        
        # Download results
        csv_results = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi",
            data=csv_results,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
        
        # Visualization
        status_counts = pd.Series(predictions).value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Distribusi Prediksi Status",
            color_discrete_map={
                'Graduate': '#90EE90',
                'Dropout': '#ff7f7f', 
                'Enrolled': '#87CEEB'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error dalam prediksi batch: {e}")

if __name__ == "__main__":
    main()