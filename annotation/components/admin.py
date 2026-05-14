"""
Admin dashboard for viewing all annotations and user progress.
"""

import streamlit as st
import pandas as pd
from utils.storage import AnnotationStorage
from utils.data_loader import DataLoader
from config import ADMIN_CREATION_KEY


def show_admin_page():
    """Admin dashboard page."""
    storage = st.session_state.storage
    current_user = st.session_state.current_user
    
    # Check if user is admin
    user = storage.get_user(current_user)
    is_admin = user and user.get('role') == 'admin'
    
    if not is_admin:
        st.error("❌ Access Denied: Only admins can access this dashboard")
        return
    
    st.title("👨‍💼 Admin Dashboard")
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "👤 User Management", "📝 Annotations", "📥 Export", "🔍 Quality Review", "⚙️ Settings"])
    
    # Tab 1: Overview
    with tab1:
        st.subheader("System Overview")
        
        # Load data
        data_loader = DataLoader()
        all_data = data_loader.load_all_data()
        
        try:
            history_df = storage.load_history_df()
        except Exception as e:
            st.error(f"Error loading history: {e}")
            history_df = pd.DataFrame()
        
        users = storage.load_users()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(all_data))
        
        with col2:
            st.metric("Total Users", len(users))
        
        with col3:
            total_annotations = len(history_df) if not history_df.empty else 0
            st.metric("Total Annotations", total_annotations)
        
        with col4:
            if not history_df.empty:
                unique_annotated = history_df['image_path'].nunique()
                st.metric("Unique Images Annotated", unique_annotated)
            else:
                st.metric("Unique Images Annotated", 0)
        
        st.markdown("---")
        
        # Folder statistics
        st.subheader("📁 Data Folder Statistics")
        folder_stats = data_loader.get_folder_stats()
        
        if folder_stats:
            stats_data = []
            for folder, stats in folder_stats.items():
                stats_data.append({
                    'Folder': folder,
                    'Images': stats['image_count'],
                    'Labels': stats['label_count'],
                    'Match': '✅' if stats['match'] else '❌'
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Warning for mismatches
            mismatches = [s for s in stats_data if s['Match'] == '❌']
            if mismatches:
                st.warning(f"⚠️ {len(mismatches)} folder(s) have mismatched image/label counts")
        else:
            st.info("No data folders found")
        
        st.markdown("---")
        
        # Annotation progress chart
        if not history_df.empty:
            st.subheader("📈 Annotation Progress Over Time")
            
            try:
                # Convert timestamp to datetime (make a copy to avoid modifying original)
                time_df = history_df.copy()
                time_df['timestamp'] = pd.to_datetime(time_df['timestamp'], errors='coerce')
                # Remove rows with invalid timestamps
                time_df = time_df.dropna(subset=['timestamp'])
                
                if not time_df.empty:
                    time_df['date'] = time_df['timestamp'].dt.date
                    
                    # Group by date
                    daily_counts = time_df.groupby('date').size().reset_index(name='count')
                    daily_counts['cumulative'] = daily_counts['count'].cumsum()
                    
                    st.line_chart(daily_counts.set_index('date')['cumulative'])
                else:
                    st.info("No valid timestamp data available")
            except Exception as e:
                st.warning(f"Could not parse timestamps: {e}")
    
    # Tab 2: User Management
    with tab2:
        st.subheader("User Management")
        
        users = storage.load_users()
        
        # Sub-tabs for user management
        umgmt_tab1, umgmt_tab2, umgmt_tab3 = st.tabs(["📋 All Users", "➕ Create User", "🔧 Manage Users"])
        
        # Tab 2.1: All Users
        with umgmt_tab1:
            st.markdown("**Registered Users**")
            
            if users:
                users_df_display = pd.DataFrame([
                    {
                        'Username': u['username'],
                        'Role': u['role'].capitalize(),
                        'Created': u.get('created_at', 'N/A')[:10],
                        'Last Login': u.get('last_login', 'Never')[:10] if u.get('last_login') else 'Never',
                        'Status': '🟢 Active' if u.get('is_active', True) else '🔴 Inactive'
                    }
                    for u in users
                ])
                
                st.dataframe(users_df_display, use_container_width=True, hide_index=True)
                
                # User statistics
                st.markdown("---")
                st.markdown("**User Statistics**")
                
                user_stats = storage.get_all_user_stats()
                stats_df = pd.DataFrame(user_stats)
                
                if not stats_df.empty:
                    stats_df_display = stats_df.rename(columns={
                        'username': 'Username',
                        'role': 'Role',
                        'total': 'Total',
                        'correct': 'Correct',
                        'incorrect': 'Incorrect',
                        'correct_percentage': 'Accuracy %'
                    })
                    
                    stats_df_display['Accuracy %'] = stats_df_display['Accuracy %'].round(1)
                    st.dataframe(stats_df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No users registered yet")
        
        # Tab 2.2: Create User
        with umgmt_tab2:
            st.markdown("**Create New User Account**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input(
                    "Username",
                    placeholder="At least 3 characters",
                    key="admin_new_username"
                )
            
            with col2:
                new_role = st.selectbox(
                    "Role",
                    ["Annotator", "Admin"],
                    key="admin_new_role"
                )
            
            new_password = st.text_input(
                "Password",
                type="password",
                placeholder="Min 8 chars, uppercase, number",
                key="admin_new_password"
            )
            
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm password",
                key="admin_confirm_password"
            )
            
            admin_key_creation = None
            if new_role == "Admin":
                admin_key_creation = st.text_input(
                    "Admin Creation Key",
                    type="password",
                    key="admin_creation_key",
                    help="Required to create admin accounts"
                )
            
            if st.button("✅ Create User", type="primary", use_container_width=True):
                if not new_username.strip():
                    st.error("⚠️ Please enter a username")
                elif new_password != confirm_password:
                    st.error("⚠️ Passwords don't match")
                else:
                    success, message = storage.register_user(
                        new_username.strip(),
                        new_password,
                        new_role.lower(),
                        admin_key_creation
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"⚠️ {message}")
        
        # Tab 2.3: Manage Users
        with umgmt_tab3:
            st.markdown("**Manage Existing Users**")
            
            if users:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Disable/Enable User**")
                    user_to_toggle = st.selectbox(
                        "Select user",
                        [u['username'] for u in users],
                        key="user_toggle"
                    )
                    
                    selected_user = next((u for u in users if u['username'] == user_to_toggle), None)
                    is_active = selected_user.get('is_active', True) if selected_user else True
                    
                    if is_active:
                        if st.button("🔒 Disable User", use_container_width=True):
                            if storage.disable_user(user_to_toggle):
                                st.success(f"User {user_to_toggle} disabled")
                                st.rerun()
                    else:
                        if st.button("🔓 Enable User", use_container_width=True):
                            if storage.enable_user(user_to_toggle):
                                st.success(f"User {user_to_toggle} enabled")
                                st.rerun()
                
                with col2:
                    st.markdown("**Reset Password**")
                    user_to_reset = st.selectbox(
                        "Select user",
                        [u['username'] for u in users],
                        key="user_reset_pwd"
                    )
                    
                    new_pwd = st.text_input(
                        "New Password",
                        type="password",
                        key="reset_new_pwd",
                        placeholder="Min 8 chars, uppercase, number"
                    )
                    
                    if st.button("🔑 Reset Password", use_container_width=True):
                        success, message = storage.update_password(user_to_reset, new_pwd)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"⚠️ {message}")
                
                with col3:
                    st.markdown("**Delete User**")
                    user_to_delete = st.selectbox(
                        "Select user",
                        [u['username'] for u in users],
                        key="user_delete"
                    )
                    
                    if st.button("🗑️ Delete User", use_container_width=True, help="Annotations will be kept"):
                        if storage.delete_user(user_to_delete):
                            st.success(f"User {user_to_delete} deleted (annotations kept)")
                            st.rerun()
            else:
                st.info("No users to manage")
    
    # Tab 3: User Statistics
    with tab3:
        st.subheader("User Statistics")
        
        user_stats = storage.get_all_user_stats()
        if user_stats:
            df = pd.DataFrame(user_stats)
            
            df_display = df.rename(columns={
                'username': 'Username',
                'role': 'Role',
                'total': 'Total',
                'correct': 'Correct',
                'incorrect': 'Incorrect',
                'correct_percentage': 'Accuracy %'
            })
            
            # Format accuracy
            df_display['Accuracy %'] = df_display['Accuracy %'].round(1)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # User comparison chart
            if len(user_stats) > 1:
                st.markdown("---")
                st.subheader("📊 User Comparison")
                
                chart_data = df[['username', 'total', 'correct', 'incorrect']].set_index('username')
                st.bar_chart(chart_data)
        else:
            st.info("No users registered yet")
    
    # Tab 4: Annotations (was Tab 3)
    with tab4:
        st.subheader("Annotation History")
        
        history_df = storage.load_history_df()
        
        if not history_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_user = st.selectbox(
                    "Filter by user:",
                    options=['All'] + sorted(history_df['annotator'].unique().tolist())
                )
            
            with col2:
                filter_correctness = st.selectbox(
                    "Filter by status:",
                    options=['All', 'Correct', 'Incorrect', 'Invalid']
                )
            
            with col3:
                filter_folder = st.selectbox(
                    "Filter by folder:",
                    options=['All'] + sorted(history_df['folder'].unique().tolist())
                )
            
            # Apply filters
            filtered_df = history_df.copy()
            
            if filter_user != 'All':
                filtered_df = filtered_df[filtered_df['annotator'] == filter_user]
            
            if filter_correctness == 'Correct':
                filtered_df = filtered_df[(filtered_df['is_correct'] == True) & (filtered_df['corrected_label'] != 'INVALID_SAMPLE')]
            elif filter_correctness == 'Incorrect':
                filtered_df = filtered_df[(filtered_df['is_correct'] == False) & (filtered_df['corrected_label'] != 'INVALID_SAMPLE')]
            elif filter_correctness == 'Invalid':
                filtered_df = filtered_df[filtered_df['corrected_label'] == 'INVALID_SAMPLE']
            
            if filter_folder != 'All':
                filtered_df = filtered_df[filtered_df['folder'] == filter_folder]
            
            # Display count
            st.write(f"Showing {len(filtered_df)} annotations")
            
            # Display dataframe
            display_df = filtered_df[[
                'annotation_id', 'folder', 'filename', 'suggested_label', 
                'is_correct', 'corrected_label', 'annotator', 'timestamp'
            ]].copy()
            
            # Format columns
            display_df['is_correct'] = display_df.apply(
                lambda row: '⚠️ Invalid' if row['corrected_label'] == 'INVALID_SAMPLE' 
                else ('✅ Correct' if row['is_correct'] else '❌ Incorrect'),
                axis=1
            )
            display_df = display_df.rename(columns={
                'annotation_id': 'ID',
                'folder': 'Folder',
                'filename': 'File',
                'suggested_label': 'Suggested',
                'is_correct': 'Status',
                'corrected_label': 'Correction',
                'annotator': 'Annotator',
                'timestamp': 'Timestamp'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Statistics for filtered data
            if len(filtered_df) > 0:
                st.markdown("---")
                st.subheader("📊 Filtered Data Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total", len(filtered_df))
                
                with col2:
                    correct_count = (filtered_df['is_correct'] == True).sum()
                    st.metric("Correct", correct_count)
                
                with col3:
                    incorrect_count = (filtered_df['is_correct'] == False).sum()
                    st.metric("Incorrect", incorrect_count)
                
                with col4:
                    accuracy = (correct_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                    st.metric("Accuracy", f"{accuracy:.1f}%")
        else:
            st.info("No annotations yet")
    
    # Tab 5: Export (was Tab 4)
    with tab5:
        st.subheader("Export Annotations")
        
        history_df = storage.load_history_df()
        
        if not history_df.empty:
            st.write("Export all annotations from all users in a single file with annotator attribution.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📄 CSV Export")
                st.write("Export as CSV (Excel compatible)")
                
                # Generate CSV
                csv_data = history_df.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Download All Annotations (CSV)",
                    data=csv_data,
                    file_name="all_annotations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("### 📋 JSON Export")
                st.write("Export as JSON (machine readable)")
                
                # Generate JSON
                json_data = history_df.to_json(orient='records', indent=2)
                
                st.download_button(
                    label="⬇️ Download All Annotations (JSON)",
                    data=json_data,
                    file_name="all_annotations.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Preview
            st.subheader("📋 Data Preview")
            st.write(f"Total annotations: {len(history_df)}")
            
            preview_df = history_df.head(10).copy()
            preview_df['is_correct'] = preview_df['is_correct'].map({True: '✅', False: '❌'})
            
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Export by user
            st.subheader("📦 Per-User Exports")
            st.write("Export annotations for individual users")
            
            users = sorted(history_df['annotator'].unique().tolist())
            
            selected_user = st.selectbox("Select user:", users)
            
            if selected_user:
                user_df = history_df[history_df['annotator'] == selected_user]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = user_df.to_csv(index=False)
                    st.download_button(
                        label=f"⬇️ Download {selected_user}'s Annotations (CSV)",
                        data=csv_data,
                        file_name=f"{selected_user}_annotations.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_data = user_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label=f"⬇️ Download {selected_user}'s Annotations (JSON)",
                        data=json_data,
                        file_name=f"{selected_user}_annotations.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                st.info(f"📊 {selected_user} has {len(user_df)} annotations")
        else:
            st.info("No annotations to export yet")
    
    # Tab 6: Quality Review (was Tab 5)
    with tab6:
        st.subheader("🔍 Quality Review")
        st.write("Identify potential annotation issues and conflicts")
        
        history_df = storage.load_history_df()
        
        if not history_df.empty:
            # Multiple annotations for same image
            st.markdown("### 🔄 Multi-Annotated Images")
            st.write("Images that have been annotated by multiple users or multiple times")
            
            multi_annotated = history_df.groupby('image_path').filter(lambda x: len(x) > 1)
            
            if not multi_annotated.empty:
                # Group by image and show conflicts
                image_groups = multi_annotated.groupby('image_path')
                
                st.write(f"Found {len(image_groups)} images with multiple annotations")
                
                # Select an image to review
                image_paths = sorted(multi_annotated['image_path'].unique())
                selected_image = st.selectbox("Select image to review:", image_paths, key="quality_review_image")
                
                if selected_image:
                    image_anns = multi_annotated[multi_annotated['image_path'] == selected_image]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Load and display image
                        data_loader = DataLoader()
                        img = data_loader.load_image(selected_image)
                        if img:
                            st.image(img, use_container_width=True)
                    
                    with col2:
                        st.markdown("**All annotations for this image:**")
                        
                        display_df = image_anns[['annotator', 'suggested_label', 'is_correct', 'corrected_label', 'timestamp']].copy()
                        display_df['is_correct'] = display_df['is_correct'].map({True: '✅ Correct', False: '❌ Incorrect'})
                        display_df = display_df.rename(columns={
                            'annotator': 'Annotator',
                            'suggested_label': 'Suggested',
                            'is_correct': 'Status',
                            'corrected_label': 'Correction',
                            'timestamp': 'Timestamp'
                        })
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Check for disagreements
                        unique_labels = set()
                        for _, row in image_anns.iterrows():
                            if row['is_correct']:
                                unique_labels.add(row['suggested_label'])
                            else:
                                unique_labels.add(row['corrected_label'])
                        
                        if len(unique_labels) > 1:
                            st.warning(f"⚠️ Disagreement detected! {len(unique_labels)} different labels: {', '.join(unique_labels)}")
                        else:
                            st.success(f"✅ All annotators agree on: '{list(unique_labels)[0]}'")
            else:
                st.info("No multi-annotated images found")
            
            st.markdown("---")
            
            # Correction rate analysis
            st.markdown("### 📊 Correction Rate Analysis")
            st.write("Images with high correction rates may indicate labeling issues")
            
            incorrect_df = history_df[history_df['is_correct'] == False]
            
            if not incorrect_df.empty:
                # Group by folder
                folder_corrections = incorrect_df.groupby('folder').size().reset_index(name='corrections')
                folder_totals = history_df.groupby('folder').size().reset_index(name='total')
                
                folder_stats = folder_corrections.merge(folder_totals, on='folder')
                folder_stats['correction_rate'] = (folder_stats['corrections'] / folder_stats['total'] * 100).round(1)
                folder_stats = folder_stats.sort_values('correction_rate', ascending=False)
                
                st.dataframe(
                    folder_stats.rename(columns={
                        'folder': 'Folder',
                        'corrections': 'Corrections',
                        'total': 'Total Annotations',
                        'correction_rate': 'Correction Rate %'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Most common corrections
                st.markdown("### 🔤 Most Common Corrections")
                
                corrections = incorrect_df.groupby(['suggested_label', 'corrected_label']).size().reset_index(name='count')
                corrections = corrections.sort_values('count', ascending=False).head(20)
                
                st.dataframe(
                    corrections.rename(columns={
                        'suggested_label': 'Original Label',
                        'corrected_label': 'Corrected To',
                        'count': 'Occurrences'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No corrections found - all annotations marked as correct!")
            
            st.markdown("---")
            
            # Inter-annotator agreement
            st.markdown("### 🤝 Inter-Annotator Agreement")
            st.write("Measure agreement between different annotators on the same images")
            
            # Find images annotated by multiple different annotators
            image_annotators = history_df.groupby('image_path')['annotator'].apply(lambda x: x.unique().tolist())
            multi_user_images = {k: v for k, v in image_annotators.items() if len(v) > 1}
            
            if multi_user_images:
                st.write(f"Found {len(multi_user_images)} images annotated by multiple users")
                
                # Calculate agreement
                agreements = 0
                disagreements = 0
                
                for image_path, annotators in multi_user_images.items():
                    image_data = history_df[history_df['image_path'] == image_path]
                    
                    # Get final labels from each annotator
                    labels = []
                    for annotator in annotators:
                        ann_data = image_data[image_data['annotator'] == annotator].iloc[-1]  # Get latest
                        if ann_data['is_correct']:
                            labels.append(ann_data['suggested_label'])
                        else:
                            labels.append(ann_data['corrected_label'])
                    
                    # Check agreement
                    if len(set(labels)) == 1:
                        agreements += 1
                    else:
                        disagreements += 1
                
                total = agreements + disagreements
                agreement_rate = (agreements / total * 100) if total > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Agreement", f"{agreement_rate:.1f}%")
                with col2:
                    st.metric("Agreements", agreements)
                with col3:
                    st.metric("Disagreements", disagreements)
                
                if disagreements > 0:
                    st.info(f"💡 Review the 'Multi-Annotated Images' section above to resolve {disagreements} disagreement(s)")
            else:
                st.info("No images have been annotated by multiple users yet")
        else:
            st.info("No annotations to review yet")
