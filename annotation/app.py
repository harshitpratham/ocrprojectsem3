"""
Word Recognition Annotation Tool
Multi-user annotation portal for word image labeling with admin dashboard.
"""

import streamlit as st
from utils.storage import AnnotationStorage

# Page configuration
st.set_page_config(
    page_title="Word Recognition Annotation Tool",
    page_icon="📝",
    layout="wide"
)

# Initialize storage
if 'storage' not in st.session_state:
    st.session_state.storage = AnnotationStorage()


def login_page():
    """Display login page with password authentication."""
    st.title("📝 Word Recognition Annotation Tool")
    st.markdown("---")
    
    # Welcome message
    st.markdown("""
    ### Welcome to the Word Recognition Annotation Tool!
    
    This tool helps you annotate word images with their correct labels. 
    
    **Features:**
    - ⚡ Fast annotation with keyboard shortcuts
    - 📊 Progress tracking and statistics
    - 👥 Multi-user support with role-based access
    - 💾 Export annotations in CSV or JSON format
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    storage = st.session_state.storage
    
    # CENTER: LOGIN ONLY
    with col2:
        st.subheader("🔓 Login")
        
        login_username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username"
        )
        
        login_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )
        
        if st.button("Login", type="primary", use_container_width=True):
            if login_username.strip() and login_password.strip():
                success, user = storage.authenticate_user(login_username.strip(), login_password)
                
                if success:
                    st.session_state.current_user = user['username']
                    st.session_state.current_role = user['role']
                    st.success(f"✅ Welcome {user['username']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.error("⚠️ Please enter both username and password")
        
        st.markdown("---")
        st.info("💡 Contact administrator to create an account")


def main():
    """Main application entry point."""
    
    # Check if user is logged in
    if 'current_user' not in st.session_state:
        login_page()
        return
    
    # User is logged in - show appropriate page based on role
    username = st.session_state.current_user
    role = st.session_state.current_role
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.title("📝 Annotation Tool")
        st.markdown(f"**User:** {username}")
        st.markdown(f"**Role:** {role}")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            del st.session_state.current_user
            del st.session_state.current_role
            st.rerun()
    
    # Route to appropriate page
    if role == "admin":
        # Import and show admin page
        from components.admin import show_admin_page
        show_admin_page()
    else:
        # Import and show annotation page
        from components.annotate import show_annotation_page
        show_annotation_page()


if __name__ == "__main__":
    main()
