import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import os

# Define the file path
# IMPORTANT: Update this path to where you save the Excel file on your system
file_path = "listado_aranceles_de_referencia_2025_20012025_ies.xlsx"

# --- Data Loading and Preprocessing (Cached) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads data from the Excel file, preprocesses it, and returns a DataFrame.
    Caches the result to avoid reloading on each Streamlit interaction.
    """
    try:
        df_sheet1 = pd.read_excel(file_path, sheet_name=0)
        df_sheet2 = pd.read_excel(file_path, sheet_name=1)
    except FileNotFoundError:
        st.error(f"Error: El archivo no se encontró en la ruta especificada: {file_path}")
        st.stop() # Stop the Streamlit app

    # 1. Combinar los dos DataFrames
    df_combined = pd.concat([df_sheet1, df_sheet2], ignore_index=True)

    # Select the relevant columns using their original names (including NOMBRE DE LA SEDE)
    required_columns = ['TIPO DE INSTITUCIÓN', 'NOMBRE_INSTITUCION', 'NOMBRE_CARRERA', 'JORNADA', 'ARANCEL ANUAL 2025', 'ARANCEL DE REFERENCIA 2025 ', 'NOMBRE DE LA SEDE']
    for col in required_columns:
        if col not in df_combined.columns:
            st.error(f"Error: Columna '{col}' no encontrada en el archivo Excel.")
            st.stop() # Stop the Streamlit app if a required column is missing

    df_combined = df_combined[required_columns]

    # 2. Limpiar los nombres de las columnas, handling accented characters
    def clean_col_name(col):
        col = col.lower()
        # Normalize accented characters
        col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf8') # Changed to utf8
        col = re.sub(r'[^\w\s]', '', col)
        col = re.sub(r'\s+', '_', col).strip()
        return col

    df_combined.columns = [clean_col_name(col) for col in df_combined.columns]

    # Rename columns to shorter, more descriptive names after cleaning
    df_combined = df_combined.rename(columns={
        'tipo_de_institucion': 'tipo_institucion',
        'nombre_institucion': 'institucion',
        'nombre_carrera': 'carrera',
        'jornada': 'modalidad',
        'arancel_anual_2025': 'arancel_anual',
        'arancel_de_referencia_2025_': 'arancel_referencia',
        'nombre_de_la_sede': 'sede' # Rename sede column as well
    })

    # 3. Convertir columnas relevantes a minúsculas
    for col in ['institucion', 'carrera', 'modalidad', 'sede']: # Include 'sede'
        df_combined[col] = df_combined[col].str.lower()

    # 4. Eliminar filas con values faltantes en las columnas clave
    df_combined.dropna(subset=['tipo_institucion', 'institucion', 'carrera', 'modalidad', 'sede', 'arancel_anual', 'arancel_referencia'], inplace=True) # Include 'sede'

    # Convert arancel columns to numeric, coercing errors
    df_combined['arancel_anual'] = pd.to_numeric(df_combined['arancel_anual'], errors='coerce')
    df_combined['arancel_referencia'] = pd.to_numeric(df_combined['arancel_referencia'], errors='coerce')

    # Drop rows where arancel values could not be converted to numeric
    df_combined.dropna(subset=['arancel_anual', 'arancel_referencia'], inplace=True)

    # 5. Eliminar filas duplicadas
    df_combined.drop_duplicates(inplace=True)

    return df_combined

# --- Flexible Search Functions (Adapted for Streamlit Context if needed, but basic functions are fine) ---
# Note: Fuzzy matching logic is typically done on filtered subsets in Streamlit based on user input
# These functions are more for finding options based on partial input if a text input was used for search

def clean_input(input_str):
    """Cleans user input string for better matching."""
    if isinstance(input_str, str):
        input_str = input_str.lower().strip()
        # Remove extra spaces
        input_str = re.sub(r'\s+', ' ', input_str)
        # Handle accented characters
        input_str = unicodedata.normalize('NFKD', input_str).encode('ascii', 'ignore').decode('utf8') # Changed to utf8
        return input_str
    return ""

# In Streamlit, direct text input with fuzzy matching and dynamic selectboxes
# would require a different approach using callbacks or managing state for suggestions.
# For this Streamilit version, we primarily use selectboxes based on available options
# filtered sequentially, which implicitly handles the "matching" based on the dropdown.

# Function to capitalize the first letter of each word for display,
# with special handling for 'cft' and 'ip'
def format_institution_type(text):
    if text is None:
        return "Seleccione uno"
    if isinstance(text, str):
        cleaned_text = clean_input(text)
        if cleaned_text == 'cft':
            return 'CFT'
        elif cleaned_text == 'ip':
            return 'IP'
        else:
            return ' '.join(word.capitalize() for word in text.split())
    return text


# Function to capitalize the first letter of each word for display (general use)
def capitalize_words(text):
    if isinstance(text, str):
        return ' '.join(word.capitalize() for word in text.split())
    return text


# --- Streamlit Application Layout and Logic ---

# Changed title from "Calculador" to "Calculadora"
st.title("Calculadora de Diferencia Arancelaria CAE 2025")

# Add Disclaimer at the beginning
st.warning("""
**Nota Importante:** Los aranceles de referencia utilizados en esta calculadora provienen de la información pública entregada por la Subsecretaría de Educación Superior. Los resultados de los cálculos son **solo referenciales y estimaciones**, y en ningún caso deben considerarse como valores definitivos o vinculantes. Para información precisa sobre tu situación particular, consulta directamente con la institución de educación superior y las autoridades competentes.
""")


st.write("Utiliza este programa para calcular la diferencia entre el arancel real y el arancel de referencia (CAE) de una carrera. Selecciona los detalles a continuación:")

# Load the data
df_combined = load_and_preprocess_data(file_path)

# Initialize session state for selections if not already present
if 'selected_tipo_institucion' not in st.session_state:
    st.session_state['selected_tipo_institucion'] = None
if 'selected_institucion' not in st.session_state:
    st.session_state['selected_institucion'] = None
if 'selected_carrera' not in st.session_state:
    st.session_state['selected_carrera'] = None
if 'selected_sede' not in st.session_state:
    st.session_state['selected_sede'] = None
if 'selected_modalidad' not in st.session_state:
    st.session_state['selected_modalidad'] = None
if 'diferencia_arancelaria_initial' not in st.session_state:
    st.session_state['diferencia_arancelaria_initial'] = None
if 'arancel_real' not in st.session_state:
    st.session_state['arancel_real'] = None
if 'arancel_cae' not in st.session_state:
    st.session_state['arancel_cae'] = None
# Initialize current_difference to None or initial difference
if 'current_difference' not in st.session_state:
     st.session_state['current_difference'] = None
# Initialize discount_applied_flag
if 'discount_applied_flag' not in st.session_state:
    st.session_state['discount_applied_flag'] = False
# Initialize state for custom tuition
if 'use_custom_tuition' not in st.session_state:
    st.session_state['use_custom_tuition'] = False
if 'custom_arancel_real' not in st.session_state:
    st.session_state['custom_arancel_real'] = 0.0


# --- User Input and Filtering (Streamlit Components) ---

st.subheader("Seleccione los detalles del programa")

# Option to input custom tuition
# Use a callback to reset selections when switching to custom tuition mode
def toggle_custom_tuition():
    # Only clear state if switching to custom tuition
    if st.session_state.get('use_custom_tuition_checkbox'): # Use .get() for safety
        st.session_state['selected_tipo_institucion'] = None
        st.session_state['selected_institucion'] = None
        st.session_state['selected_carrera'] = None
        st.session_state['selected_sede'] = None
        st.session_state['selected_modalidad'] = None
        st.session_state['arancel_real'] = None
        st.session_state['arancel_cae'] = None
        st.session_state['diferencia_arancelaria_initial'] = None
        st.session_state['current_difference'] = None
        st.session_state['discount_applied_flag'] = False
        # Keep custom arancel value if already entered, no need to reset it here

st.session_state['use_custom_tuition'] = st.checkbox(
    "¿Conoces tu arancel real? Ingresa un valor personalizado.",
    value=st.session_state['use_custom_tuition'],
    key='use_custom_tuition_checkbox',
    on_change=toggle_custom_tuition # Use callback
)

arancel_real = None
arancel_cae = None


# Determine arancel_real and arancel_cae based on mode and selection
if st.session_state.get('use_custom_tuition'): # Use .get()
    st.session_state['custom_arancel_real'] = st.number_input(
        "Ingrese su arancel real personalizado:",
        min_value=0.0,
        step=1000.0,
        format="%.0f",
        key='custom_arancel_real_input',
        help="Ingrese el arancel real anual de su carrera."
    )
    arancel_real = st.session_state.get('custom_arancel_real') # Use .get() for safety
    # Ensure arancel_real is treated as numeric if not None
    if arancel_real is not None:
        try:
            arancel_real = float(arancel_real)
        except ValueError:
             arancel_real = None # Handle potential conversion errors


    # If using custom tuition, still need CAE from selected program if available
    # Show selectors even in custom mode to allow selecting a program for CAE
    # Step 1: Institution Type Selection
    available_tipos = df_combined['tipo_institucion'].unique().tolist()
    selected_tipo_institucion = st.selectbox(
        "Tipo de institución (para obtener Arancel CAE):", # Adjusted label
        options=[None] + available_tipos, # Add None as initial option
        index=available_tipos.index(st.session_state.get('selected_tipo_institucion')) + 1 if st.session_state.get('selected_tipo_institucion') in available_tipos else 0, # Use .get()
        format_func=format_institution_type, # Use custom format function
        key='selected_tipo_institucion',
        help="Seleccione el tipo de institución para obtener el Arancel de Referencia (CAE)."
    )

    # Filter based on Institution Type
    df_tipo_filtered = pd.DataFrame()
    if st.session_state.get('selected_tipo_institucion'): # Use .get()
        df_tipo_filtered = df_combined[df_combined['tipo_institucion'] == st.session_state['selected_tipo_institucion']].copy()
        available_instituciones = df_tipo_filtered['institucion'].unique().tolist()

        # Step 2: Institution Name Selection
        selected_institucion = st.selectbox(
            "Nombre de la institución (para obtener Arancel CAE):", # Adjusted label
            options=[None] + available_instituciones,
             index=available_instituciones.index(st.session_state.get('selected_institucion')) + 1 if st.session_state.get('selected_institucion') in available_instituciones else 0, # Use .get()
             format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
             key='selected_institucion',
            help=f"Seleccione la institución ({capitalize_words(st.session_state.get('selected_tipo_institucion', ''))})." # Use .get()
        )

        # Filter based on Institution Name
        df_institucion_filtered = pd.DataFrame()
        if st.session_state.get('selected_institucion'): # Use .get()
            df_institucion_filtered = df_tipo_filtered[df_tipo_filtered['institucion'] == st.session_state['selected_institucion']].copy()
            available_carreras = df_institucion_filtered['carrera'].unique().tolist()

            # Step 3: Program Name Selection
            selected_carrera = st.selectbox(
                "Nombre de la carrera (para obtener Arancel CAE):", # Adjusted label
                options=[None] + available_carreras,
                 index=available_carreras.index(st.session_state.get('selected_carrera')) + 1 if st.session_state.get('selected_carrera') in available_carreras else 0, # Use .get()
                 format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
                 key='selected_carrera',
                help=f"Seleccione la carrera en {capitalize_words(st.session_state.get('selected_institucion', ''))}." # Use .get()
            )

            # Filter based on Program Name
            df_carrera_filtered = pd.DataFrame()
            if st.session_state.get('selected_carrera'): # Use .get()
                df_carrera_filtered = df_institucion_filtered[df_institucion_filtered['carrera'] == st.session_state['selected_carrera']].copy()
                available_sedes = df_carrera_filtered['sede'].unique().tolist()

                # Step 4: Sede Selection
                selected_sede = st.selectbox(
                    "Nombre de la sede (para obtener Arancel CAE):", # Adjusted label
                    options=[None] + available_sedes,
                    index=available_sedes.index(st.session_state.get('selected_sede')) + 1 if st.session_state.get('selected_sede') in available_sedes else 0, # Use .get()
                    format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
                    key='selected_sede',
                     help=f"Seleccione la sede para {capitalize_words(st.session_state.get('selected_carrera', ''))}." # Use .get()
                )

                # Filter based on Sede
                df_sede_filtered = pd.DataFrame()
                if st.session_state.get('selected_sede'): # Use .get()
                    df_sede_filtered = df_carrera_filtered[df_carrera_filtered['sede'] == st.session_state['selected_sede']].copy()
                    available_modalidades = df_sede_filtered['modalidad'].unique().tolist()

                    # Step 5: Modality Selection
                    selected_modalidad = st.selectbox(
                        "Modalidad/Jornada (para obtener Arancel CAE):", # Adjusted label
                        options=[None] + available_modalidades,
                        index=available_modalidades.index(st.session_state.get('selected_modalidad')) + 1 if st.session_state.get('selected_modalidad') in available_modalidades else 0, # Use .get()
                        format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
                        key='selected_modalidad',
                        help=f"Seleccione la modalidad para {capitalize_words(st.session_state.get('selected_carrera', ''))} en {capitalize_words(st.session_state.get('selected_sede', ''))}." # Use .get()
                    )

                    # Final Filter to get the selected row for CAE - Only populate if selection is complete
                    if st.session_state.get('selected_modalidad'): # Use .get()
                         df_final_selection_for_cae = df_combined[
                            (df_combined['tipo_institucion'] == st.session_state.get('selected_tipo_institucion')) &
                            (df_combined['institucion'] == st.session_state.get('selected_institucion')) &
                            (df_combined['carrera'] == st.session_state.get('selected_carrera')) &
                            (df_combined['sede'] == st.session_state.get('selected_sede')) &
                            (df_combined['modalidad'] == st.session_state.get('selected_modalidad'))
                         ]
                         if not df_final_selection_for_cae.empty:
                              arancel_cae = df_final_selection_for_cae.iloc[0]['arancel_referencia']
                         else:
                              arancel_cae = None # Cannot find CAE for this combination
                    else:
                        arancel_cae = None # No program selected yet

else: # Not using custom tuition, show selectors and get both aranceles from data
    # Step 1: Institution Type Selection
    available_tipos = df_combined['tipo_institucion'].unique().tolist()
    selected_tipo_institucion = st.selectbox(
        "Tipo de institución:",
        options=[None] + available_tipos, # Add None as initial option
        index=available_tipos.index(st.session_state.get('selected_tipo_institucion')) + 1 if st.session_state.get('selected_tipo_institucion') in available_tipos else 0, # Use .get()
        format_func=format_institution_type, # Use custom format function
        key='selected_tipo_institucion',
        help="Seleccione el tipo de institución (Universidad, IP, o CFT)."
    )

    # Filter based on Institution Type
    df_tipo_filtered = pd.DataFrame()
    if st.session_state.get('selected_tipo_institucion'): # Use .get()
        df_tipo_filtered = df_combined[df_combined['tipo_institucion'] == st.session_state['selected_tipo_institucion']].copy()
        available_instituciones = df_tipo_filtered['institucion'].unique().tolist()

        # Step 2: Institution Name Selection
        selected_institucion = st.selectbox(
            "Nombre de la institución:",
            options=[None] + available_instituciones,
             index=available_instituciones.index(st.session_state.get('selected_institucion')) + 1 if st.session_state.get('selected_institucion') in available_instituciones else 0, # Use .get()
             format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
             key='selected_institucion',
            help=f"Seleccione la institución ({capitalize_words(st.session_state.get('selected_tipo_institucion', ''))})." # Use .get()
        )

        # Filter based on Institution Name
        df_institucion_filtered = pd.DataFrame()
        if st.session_state.get('selected_institucion'): # Use .get()
            df_institucion_filtered = df_tipo_filtered[df_tipo_filtered['institucion'] == st.session_state['selected_institucion']].copy()
            available_carreras = df_institucion_filtered['carrera'].unique().tolist()

            # Step 3: Program Name Selection
            selected_carrera = st.selectbox(
                "Nombre de la carrera:",
                options=[None] + available_carreras,
                 index=available_carreras.index(st.session_state.get('selected_carrera')) + 1 if st.session_state.get('selected_carrera') in available_carreras else 0, # Use .get()
                 format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
                 key='selected_carrera',
                help=f"Seleccione la carrera en {capitalize_words(st.session_state.get('selected_institucion', ''))}." # Use .get()
            )

            # Filter based on Program Name
            df_carrera_filtered = pd.DataFrame()
            if st.session_state.get('selected_carrera'): # Use .get()
                df_carrera_filtered = df_institucion_filtered[df_institucion_filtered['carrera'] == st.session_state['selected_carrera']].copy()
                available_sedes = df_carrera_filtered['sede'].unique().tolist()

                # Step 4: Sede Selection
                selected_sede = st.selectbox(
                    "Nombre de la sede:",
                    options=[None] + available_sedes,
                    index=available_sedes.index(st.session_state.get('selected_sede')) + 1 if st.session_state.get('selected_sede') in available_sedes else 0, # Use .get()
                    format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
                    key='selected_sede',
                     help=f"Seleccione la sede para {capitalize_words(st.session_state.get('selected_carrera', ''))}." # Use .get()
                )

                # Filter based on Sede
                df_sede_filtered = pd.DataFrame()
                if st.session_state.get('selected_sede'): # Use .get()
                    df_sede_filtered = df_carrera_filtered[df_carrera_filtered['sede'] == st.session_state['selected_sede']].copy()
                    available_modalidades = df_sede_filtered['modalidad'].unique().tolist()

                    # Step 5: Modality Selection
                    selected_modalidad = st.selectbox(
                        "Modalidad/Jornada:",
                        options=[None] + available_modalidades,
                        index=available_modalidades.index(st.session_state.get('selected_modalidad')) + 1 if st.session_state.get('selected_modalidad') in available_modalidades else 0, # Use .get()
                        format_func=lambda x: "Seleccione uno" if x is None else capitalize_words(x),
                        key='selected_modalidad',
                        help=f"Seleccione la modalidad para {capitalize_words(st.session_state.get('selected_carrera', ''))} en {capitalize_words(st.session_state.get('selected_sede', ''))}." # Use .get()
                    )

                    # Final Filter to get the selected row - Only populate if selection is complete
                    if st.session_state.get('selected_modalidad'): # Use .get()
                        df_final_selection = df_sede_filtered[df_sede_filtered['modalidad'] == st.session_state['selected_modalidad']].copy()
                        if not df_final_selection.empty:
                             arancel_real = df_final_selection.iloc[0]['arancel_anual']
                             arancel_cae = df_final_selection.iloc[0]['arancel_referencia']
                        else:
                             arancel_real = None
                             arancel_cae = None


# Store current arancel_real and arancel_cae in session state after determining
st.session_state['arancel_real'] = arancel_real
st.session_state['arancel_cae'] = arancel_cae

# Calculate initial difference and current difference ONLY if both aranceles are available and not None
if st.session_state.get('arancel_real') is not None and st.session_state.get('arancel_cae') is not None:
     # Ensure arancel_real and arancel_cae are treated as numeric before calculating difference
     try:
         arancel_real_numeric = float(st.session_state['arancel_real'])
         arancel_cae_numeric = float(st.session_state['arancel_cae'])

         diferencia_arancelaria_initial = arancel_real_numeric - arancel_cae_numeric
         st.session_state['diferencia_arancelaria_initial'] = diferencia_arancelaria_initial

         # Initialize or reset current_difference to initial difference if no discount applied yet or selection changed
         # Check if initial_diff_at_last_discount exists and is not None before comparison
         initial_diff_at_last_discount = st.session_state.get('initial_diff_at_last_discount')
         if 'current_difference' not in st.session_state or \
            st.session_state.get('discount_applied_flag', False) is False or \
            (initial_diff_at_last_discount is not None and st.session_state.get('diferencia_arancelaria_initial') != initial_diff_at_last_discount):

             st.session_state['current_difference'] = st.session_state.get('diferencia_arancelaria_initial') # Use .get()
             st.session_state['discount_applied_flag'] = False # Reset flag
             st.session_state['initial_diff_at_last_discount'] = st.session_state.get('diferencia_arancelaria_initial') # Store initial diff for comparison, Use .get()

         # --- Display Initial Results ---
         st.subheader("Resultados Iniciales:")
         st.write(f"Arancel Real: {st.session_state.get('arancel_real'):,.0f} CLP") # Use .get()
         st.write(f"Arancel CAE (Referencia): {st.session_state.get('arancel_cae'):,.0f} CLP") # Use .get()
         st.write(f"Diferencia Arancelaria Inicial: {st.session_state.get('diferencia_arancelaria_initial'):,.0f} CLP") # Use .get()

     except (ValueError, TypeError):
         # Handle cases where arancel_real or arancel_cae are not numeric despite being not None
         st.error("Error procesando valores de arancel. Asegúrate de que los valores del archivo sean numéricos.")
         st.session_state['diferencia_arancelaria_initial'] = None
         st.session_state['current_difference'] = None
         st.session_state['discount_applied_flag'] = False


else:
    # Reset relevant session state if arancel values are not available
    st.session_state['diferencia_arancelaria_initial'] = None
    st.session_state['current_difference'] = None
    st.session_state['discount_applied_flag'] = False
    # Clear discount display state variables
    for key in ['descuento_aplicado_monto', 'discount_display_type', 'discount_display_value', 'arancel_real_con_descuento']:
        if key in st.session_state:
            del st.session_state[key]


# --- Apply Discount (Optional) ---
# Offer discount only if initial difference is positive and we have valid arancel values
# Explicitly check that diferencia_arancelaria_initial is NOT None before comparing to 0
initial_diff_for_discount = st.session_state.get('diferencia_arancelaria_initial')
if initial_diff_for_discount is not None and initial_diff_for_discount > 0 and st.session_state.get('arancel_real') is not None and st.session_state.get('arancel_cae') is not None:
    st.subheader("Aplicar Descuento (Opcional)")
    # Use a unique key for the checkbox
    # Set default value based on session state to persist checked state
    apply_discount = st.checkbox(
        "¿Desea aplicar un descuento al arancel real?",
        value=st.session_state.get('apply_discount_checkbox_state', False),
        key='apply_discount_checkbox',
        help="Seleccione para aplicar un descuento."
    )
    # Update session state with current checkbox state
    st.session_state['apply_discount_checkbox_state'] = apply_discount


    if apply_discount:
        discount_type = st.radio(
            "Tipo de descuento:",
            options=["Monto Fijo", "Porcentaje"],
            horizontal=True,
            key='discount_type_radio'
        )

        discount_value = st.number_input(
            "Valor del descuento:",
            min_value=0.0,
            step=1.0,
            format="%.2f" if discount_type == "Porcentaje" else "%.0f",
            key='discount_value_input',
            help="Ingrese el monto o porcentaje de descuento."
        )

        # Add a button to apply the discount and recalculate
        # Use a callback function for the button to ensure state update happens on click
        def apply_discount_callback():
            arancel_real_original = st.session_state.get('arancel_real', 0.0) # Use get with default
            arancel_cae = st.session_state.get('arancel_cae', 0.0) # Still need CAE for calculation, use get with default
            discount_type = st.session_state.get('discount_type_radio', 'Monto Fijo') # Use get with default
            discount_value = st.session_state.get('discount_value_input', 0.0) # Use get with default

            # Ensure arancel_real_original is numeric before calculating discount
            try:
                arancel_real_original_numeric = float(arancel_real_original)

                descuento_aplicado = 0.0
                if discount_type == "Porcentaje":
                    if 0 <= discount_value <= 100:
                        descuento_aplicado = arancel_real_original_numeric * (discount_value / 100)
                    else:
                        st.error("Porcentaje de descuento inválido. Debe estar entre 0 y 100.")
                        descuento_aplicado = 0.0 # Reset applied discount if invalid

                elif discount_type == "Monto Fijo":
                    if discount_value >= 0:
                        descuento_aplicado = discount_value
                    else:
                        st.error("El monto de descuento no puede ser negativo.")
                        descuento_aplicado = 0.0 # Reset applied discount if invalid

                arancel_real_con_descuento = arancel_real_original_numeric - descuento_aplicado
                arancel_real_con_descuento = max(0, arancel_real_con_descuento)

                # Ensure arancel_cae is numeric before calculating new difference
                try:
                    arancel_cae_numeric = float(arancel_cae)
                    diferencia_arancelaria_con_descuento = arancel_real_con_descuento - arancel_cae_numeric

                    # Update the difference in session state for installment calculation
                    st.session_state['current_difference'] = diferencia_arancelaria_con_descuento # Store the current difference
                    st.session_state['discount_applied_flag'] = True # Use a flag to indicate discount was applied
                    st.session_state['descuento_aplicado_monto'] = descuento_aplicado # Store applied discount amount for display
                    st.session_state['discount_display_type'] = discount_type # Store type for display
                    st.session_state['discount_display_value'] = discount_value # Store value for display
                    st.session_state['arancel_real_con_descuento'] = arancel_real_con_descuento # Store discounted real tuition

                except (ValueError, TypeError):
                     st.error("Error procesando Arancel CAE para el cálculo de descuento.")
                     # Keep previous state or reset relevant states on error
                     st.session_state['discount_applied_flag'] = False # Reset flag

            except (ValueError, TypeError):
                 st.error("Error procesando Arancel Real para el cálculo de descuento.")
                 # Keep previous state or reset relevant states on error
                 st.session_state['discount_applied_flag'] = False # Reset flag


        st.button("Aplicar Descuento", key='apply_discount_button', on_click=apply_discount_callback)

        # If discount was applied in a previous run, display the results persistently
        if st.session_state.get('discount_applied_flag', False):
             st.subheader("Resultados con Descuento Aplicado:")
             st.write(f"Arancel Real Original: {st.session_state.get('arancel_real'):,.0f} CLP") # Use .get()
             if st.session_state.get('discount_display_type') == "Porcentaje":
                  st.write(f"Descuento Aplicado: {st.session_state.get('discount_display_value',0):.2f}% ({st.session_state.get('descuento_aplicado_monto',0):,.0f} CLP)")
             else:
                  st.write(f"Descuento Aplicado: {st.session_state.get('descuento_aplicado_monto',0):,.0f} CLP")
             st.write(f"Nuevo Arancel Real (con descuento): {st.session_state.get('arancel_real_con_descuento', st.session_state.get('arancel_real')):,.0f} CLP") # Use .get()
             st.write(f"Nueva Diferencia Arancelaria: {st.session_state.get('current_difference'):,.0f} CLP") # Use .get()


    # If discount checkbox is unchecked after being checked, revert the difference
    # This logic needs to happen on rerun if the checkbox state changes
    if st.session_state.get('discount_applied_flag', False) and not apply_discount:
        # Revert to the initial difference
        st.session_state['current_difference'] = st.session_state.get('diferencia_arancelaria_initial') # Use .get()
        st.session_state['discount_applied_flag'] = False # Remove the flag
        # Clear display state variables related to discount results
        for key in ['descuento_aplicado_monto', 'discount_display_type', 'discount_display_value', 'arancel_real_con_descuento']:
            if key in st.session_state:
                del st.session_state[key]


# --- Offer Installment Calculation ---
# Use the current_difference which reflects the initial or discounted value
# Only show installment calculation if there's a positive current difference AND valid arancel values
current_difference_for_installment = st.session_state.get('current_difference')
if current_difference_for_installment is not None and current_difference_for_installment > 0 and st.session_state.get('arancel_real') is not None and st.session_state.get('arancel_cae') is not None:
    st.subheader("Calcular Cuotas")
    # Set default value based on session state to persist checked state
    calculate_installments = st.checkbox(
        "¿Desea calcular las cuotas a pagar?",
        value=st.session_state.get('calculate_installments_checkbox_state', False),
        key='calculate_installments_checkbox'
    )
     # Update session state with current checkbox state
    st.session_state['calculate_installments_checkbox_state'] = calculate_installments


    if calculate_installments:
        num_cuotas = st.number_input(
            "Número de cuotas:",
            min_value=1,
            step=1,
            value=st.session_state.get('num_cuotas_input_value', 1), # Default to 1 installment, persist value
            key='num_cuotas_input',
            help="Ingrese el número de cuotas para calcular (un número entero positivo)."
        )
        # Update session state with current number input value
        st.session_state['num_cuotas_input_value'] = num_cuotas

        # Add optional interest input
        include_interest = st.checkbox(
            "Incluir interés compuesto mensual?",
            value=st.session_state.get('include_interest_checkbox_state', False),
            key='include_interest_checkbox'
        )
        st.session_state['include_interest_checkbox_state'] = include_interest

        monthly_interest_rate = 0.0
        if include_interest:
            # Input for monthly interest rate (as percentage)
            monthly_interest_rate = st.number_input(
                "Tasa de interés mensual (%)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key='monthly_interest_rate_input',
                help="Ingrese la tasa de interés mensual en porcentaje (ej: 1.5 para 1.5%)."
            )
            # Convert percentage to decimal
            monthly_interest_rate = monthly_interest_rate / 100.0


        # The installment calculation should happen whenever num_cuotas or current_difference changes
        # Ensure current_difference is not None before division
        if num_cuotas is not None and num_cuotas > 0 and st.session_state.get('current_difference') is not None:
            total_difference = st.session_state.get('current_difference') # Use .get() for safety

            # Ensure total_difference is numeric
            try:
                total_difference_numeric = float(total_difference)

                if include_interest and monthly_interest_rate is not None and monthly_interest_rate > 0:
                    # Calculate monthly payment using the compound interest formula (Amortization formula)
                    # M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
                    # M = Monthly Payment, P = Principal Loan Amount, i = monthly interest rate, n = number of payments
                    try:
                        # Ensure monthly_interest_rate is not zero in the denominator
                        if monthly_interest_rate > 0:
                             # Avoid division by zero if (1 + i)^n - 1 is very close to zero for small rates/large n
                             denominator = ((1 + monthly_interest_rate)**num_cuotas) - 1
                             if denominator != 0:
                                monto_por_cuota = total_difference_numeric * (monthly_interest_rate * (1 + monthly_interest_rate)**num_cuotas) / denominator
                                st.write(f"La diferencia total a financiar es: {total_difference_numeric:,.0f} CLP.")
                                st.write(f"El monto estimado por cuota ({num_cuotas} cuotas) con un interés mensual del {st.session_state.get('monthly_interest_rate_input', 0.0):.2f}% sería: {monto_por_cuota:,.0f} CLP.")
                                st.info("Este cálculo con interés es una estimación. Las condiciones reales de financiamiento pueden variar.")
                             else:
                                # Handle case where denominator is zero (e.g., rate is zero) - fallback to simple division
                                monto_por_cuota = total_difference_numeric / num_cuotas
                                st.write(f"La diferencia total a pagar es: {total_difference_numeric:,.0f} CLP.")
                                st.write(f"El monto por cuota ({num_cuotas} cuotas) sería: {monto_por_cuota:,.0f} CLP.")

                        else:
                            # If interest rate is 0 but checkbox is checked, revert to simple division
                            monto_por_cuota = total_difference_numeric / num_cuotas
                            st.write(f"La diferencia total a pagar es: {total_difference_numeric:,.0f} CLP.")
                            st.write(f"El monto por cuota ({num_cuotas} cuotas) sería: {monto_por_cuota:,.0f} CLP.")

                    except Exception as e:
                        st.error(f"Error al calcular cuotas con interés: {e}")
                        # Fallback to simple calculation or display error
                        monto_por_cuota = total_difference_numeric / num_cuotas
                        st.write(f"La diferencia total a pagar es: {total_difference_numeric:,.0f} CLP.")
                        st.write(f"El monto por cuota ({num_cuotas} cuotas) sería: {monto_por_cuota:,.0f} CLP.")


                else:
                    # Simple division if interest is not included or rate is 0
                    monto_por_cuota = total_difference_numeric / num_cuotas
                    st.write(f"La diferencia total a pagar es: {total_difference_numeric:,.0f} CLP.")
                    st.write(f"El monto por cuota ({num_cuotas} cuotas) sería: {monto_por_cuota:,.0f} CLP.")

            except (ValueError, TypeError):
                 st.error("Error procesando la diferencia arancelaria para el cálculo de cuotas.")


        else:
             st.error("Por favor ingrese un número de cuotas válido (entero positivo) y asegúrese de que la diferencia arancelaria sea válida.")


# Display message if CAE covers tuition
# Check if a full selection has been made OR custom tuition is used AND the current difference is <= 0
# Also ensure arancel_real and arancel_cae are available, and current_difference is not None for comparison
if st.session_state.get('arancel_real') is not None and st.session_state.get('arancel_cae') is not None:
    # Only display if aranceles are set, then check current_difference
    current_difference_for_cover_check = st.session_state.get('current_difference')
    if current_difference_for_cover_check is not None and current_difference_for_cover_check <= 0: # Use .get()
         st.info("¡Buenas noticias! El arancel de referencia (CAE) cubre completamente el arancel real (considerando cualquier descuento aplicado) para la selección/valor ingresado.")

# Display messages prompting user to complete selection or provide custom input
if st.session_state.get('arancel_real') is None or st.session_state.get('arancel_cae') is None:
    if not st.session_state.get('use_custom_tuition'): # Use .get()
         if st.session_state.get('selected_modalidad') is None: # Use .get()
              st.info("Por favor, complete la selección de los detalles del programa para ver los resultados iniciales.")
         # The elif below is no longer strictly needed here as the arancel check above covers it,
         # but keep for clarity if arancel_real/cae somehow become None after modalidad is selected.
         # elif st.session_state.get('arancel_real') is not None or st.session_state.get('arancel_cae') is not None:
         #      st.warning("No se encontraron datos completos para la selección realizada. Por favor, revise los detalles o intente otra selección.")
    elif st.session_state.get('use_custom_tuition') and st.session_state.get('arancel_cae') is None: # Use .get()
         st.info("Por favor, ingrese su arancel real y seleccione un programa para obtener el arancel de referencia (CAE) si desea calcular la diferencia.")


# Add Disclaimer near results if results are displayed (aranceles are available)
if st.session_state.get('arancel_real') is not None and st.session_state.get('arancel_cae') is not None:
    st.warning("""
    **Recuerda:** Los valores mostrados son estimaciones basadas en el Arancel de Referencia CAE 2025. El monto final a pagar y las condiciones de financiamiento pueden variar. Consulta siempre la información oficial de tu institución y el Ministerio de Educación.
    """)