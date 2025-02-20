import streamlit as st

# Initialize session state for selectbox if not already set
if 'selected_value' not in st.session_state:
    st.session_state.selected_value = 'Option 1'  # Default selection

# List of options
options = ['Option 1', 'Option 2', 'Option 3', 'Option 4']

# Function to handle selection change
def on_select_change():
    selected_value = st.session_state.selectbox_value  # This is a string
    selected_index = options.index(selected_value)     # Convert to index
    st.session_state.selected_index = selected_index   # Store index if needed
    st.write(f"Selected option: {selected_value}")
    st.success(f"Index changed to: {selected_index}")

# Selectbox with on_change callback
st.selectbox(
    'Choose an option:',
    options,
    index=options.index(st.session_state.selected_value),
    key='selectbox_value',  # Stores the selected value (not index)
    on_change=on_select_change
)

# Show current selection details
st.write(f"Current Selected Value: {st.session_state.selectbox_value}")
st.write(f"Current Selected Index: {options.index(st.session_state.selectbox_value)}")
