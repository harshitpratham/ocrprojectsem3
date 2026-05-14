"""
Annotation page for word image labeling.
Displays images with suggested labels and allows users to mark correct/incorrect.
"""

import streamlit as st
import streamlit.components.v1 as components
from utils.data_loader import DataLoader
from utils.storage import AnnotationStorage
from components.confidence_tag import confidence_tag_input


def inject_keyboard_shortcuts():
    """Inject JavaScript for keyboard shortcuts."""
    keyboard_js = """
    <script>
    const doc = window.parent.document;
    
    // Remove existing listener if any
    if (window.keyboardListenerAdded) {
        return;
    }
    window.keyboardListenerAdded = true;
    
    doc.addEventListener('keydown', function(e) {
        // Get the target element
        const target = e.target;
        const isTextInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
        
        // Enter key - Mark correct and next (only if not in text input)
        if (e.key === 'Enter' && !isTextInput) {
            e.preventDefault();
            const correctBtn = doc.querySelector('button[kind="primary"]');
            if (correctBtn && correctBtn.textContent.includes('Correct')) {
                correctBtn.click();
            }
        }
        
        // Ctrl/Cmd + Enter - Submit correction and next (when in text input)
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && isTextInput) {
            e.preventDefault();
            const submitBtn = doc.querySelector('button[kind="primary"]');
            if (submitBtn && submitBtn.textContent.includes('Submit')) {
                submitBtn.click();
            }
        }
        
        // Arrow Left - Previous (only if not in text input)
        if (e.key === 'ArrowLeft' && !isTextInput) {
            e.preventDefault();
            const prevBtn = Array.from(doc.querySelectorAll('button')).find(
                btn => btn.textContent.includes('Previous')
            );
            if (prevBtn && !prevBtn.disabled) {
                prevBtn.click();
            }
        }
        
        // Arrow Right - Next (only if not in text input)
        if (e.key === 'ArrowRight' && !isTextInput) {
            e.preventDefault();
            const nextBtn = Array.from(doc.querySelectorAll('button')).find(
                btn => btn.textContent.includes('Next')
            );
            if (nextBtn && !nextBtn.disabled) {
                nextBtn.click();
            }
        }
        
        // Backspace - Mark incorrect (only if not in text input)
        if (e.key === 'Backspace' && !isTextInput) {
            e.preventDefault();
            const incorrectRadio = doc.querySelector('input[type="radio"][value="incorrect"]');
            if (incorrectRadio) {
                incorrectRadio.click();
                // Focus on correction input after a short delay
                setTimeout(() => {
                    const correctionInput = doc.querySelector('input[placeholder="Enter correct word"]');
                    if (correctionInput) {
                        correctionInput.focus();
                    }
                }, 100);
            }
        }
    });
    </script>
    """
    components.html(keyboard_js, height=0)


def show_annotation_page():
    """Main annotation page."""
    username = st.session_state.current_user
    storage = st.session_state.storage
    
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    data_loader = st.session_state.data_loader
    
    # Load all data
    if 'all_data' not in st.session_state:
        st.session_state.all_data = data_loader.load_all_data()
    
    all_data = st.session_state.all_data
    
    # Check if data is available
    if not all_data:
        st.error("⚠️ No data found! Please ensure that:")
        st.markdown("""
        - `sorted_crops/` folder contains subfolders with images
        - `ground_truth/` folder contains corresponding .txt files
        
        Example structure:
        ```
        sorted_crops/
          31/
            000.jpg
            001.jpg
        ground_truth/
          31.txt
        ```
        """)
        return
    
    # Initialize current index
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Get annotated images for this user
    annotated_images = storage.get_annotated_images(username)
    
    # Sidebar with stats and filters
    with st.sidebar:
        st.markdown("### 📊 Your Progress")
        
        # Get user stats
        stats = storage.get_user_stats(username)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats['total'])
            st.metric("Correct", stats['correct'])
        with col2:
            st.metric("Remaining", len(all_data) - len(annotated_images))
            st.metric("Incorrect", stats['incorrect'])
        
        if stats['total'] > 0:
            st.metric("Accuracy", f"{stats['correct_percentage']:.1f}%")
        
        st.markdown("---")
        
        # Filter options
        st.markdown("### 🔍 Filters")
        show_only_unannotated = st.checkbox("Show only unannotated", value=False)
        show_only_incorrect = st.checkbox("Show only incorrect", value=False)
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export")
        
        col1, col2 = st.columns(2)
        with col1:
            if stats['total'] > 0:
                df = storage.get_user_annotations_df(username)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    file_name=f"{username}_annotations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button("📄 CSV", disabled=True, use_container_width=True)
        
        with col2:
            if stats['total'] > 0:
                import json
                annotations = storage.get_user_annotations(username)
                json_data = json.dumps(annotations, indent=2)
                st.download_button(
                    "📋 Download JSON",
                    json_data,
                    file_name=f"{username}_annotations.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.button("📋 JSON", disabled=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ⌨️ Keyboard Shortcuts")
        st.markdown("""
        - **Enter**: Mark correct & next
        - **Backspace**: Mark incorrect
        - **Ctrl+Enter**: Submit correction
        - **← →**: Navigate images
        """)
    
    # Apply filters
    filtered_data = all_data.copy()
    
    if show_only_unannotated:
        filtered_data = [d for d in filtered_data if d['image_path'] not in annotated_images]
    
    if show_only_incorrect:
        # Get user's incorrect annotations
        user_annotations = storage.get_user_annotations(username)
        incorrect_images = {ann['image_path'] for ann in user_annotations if not ann['is_correct']}
        filtered_data = [d for d in filtered_data if d['image_path'] in incorrect_images]
    
    if not filtered_data:
        st.info("🎉 No images match the current filter!")
        return
    
    # Ensure current index is valid for filtered data
    if st.session_state.current_index >= len(filtered_data):
        st.session_state.current_index = 0
    
    # Get current image data
    current_data = filtered_data[st.session_state.current_index]
    
    # Inject keyboard shortcuts
    inject_keyboard_shortcuts()
    
    # Main annotation interface
    st.title("🖼️ Word Image Annotation")
    
    # Show completion message if all done
    if len(annotated_images) >= len(all_data) and not show_only_incorrect:
        st.success("🎉 Congratulations! You've annotated all images!")
        st.balloons()
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔄 Review Incorrect Annotations", use_container_width=True):
                st.session_state.show_only_incorrect = True
                st.rerun()
    
    # Progress bar
    progress = st.session_state.current_index / len(filtered_data) if filtered_data else 0
    st.progress(progress, text=f"Image {st.session_state.current_index + 1} of {len(filtered_data)}")
    
    # Image display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Image")
        image = data_loader.load_image(current_data['image_path'])
        if image:
            st.image(image, use_container_width=True)
        else:
            st.error("Failed to load image")
        
        # Image metadata
        st.caption(f"📁 Folder: {current_data['folder']} | 📄 File: {current_data['filename']}")
    
    with col2:
        st.markdown("### Suggested Label")
        st.markdown(f"## `{current_data['suggested_label']}`")
        
        # Check if already annotated
        latest_annotation = storage.get_latest_annotation_for_image(
            current_data['image_path'], 
            username
        )
        
        if latest_annotation:
            st.info(f"✅ Already annotated as: {'Correct' if latest_annotation['is_correct'] else 'Incorrect'}")
            if not latest_annotation['is_correct']:
                st.write(f"Correction: `{latest_annotation['corrected_label']}`")
        
        st.markdown("---")
        st.markdown("### Review Image")
        
        # Annotation form
        is_correct = st.radio(
            "Select one:",
            options=["correct", "incorrect", "invalid"],
            format_func=lambda x: "✅ Correct" if x == "correct" else ("❌ Incorrect" if x == "incorrect" else "⚠️ Invalid Sample"),
            key=f"radio_{st.session_state.current_index}"
        )
        
        corrected_label = ""
        if is_correct == "incorrect":
            corrected_label = st.text_input(
                "Enter correct word:",
                placeholder="Enter correct word",
                key=f"correction_{st.session_state.current_index}"
            )

        confidence_tag = confidence_tag_input(key_prefix=f"i{st.session_state.current_index}")

        # Submit buttons
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if is_correct == "correct":
                if st.button("✅ Correct & Next", type="primary", use_container_width=True):
                    # Save annotation
                    storage.save_annotation({
                        'image_path': current_data['image_path'],
                        'folder': current_data['folder'],
                        'filename': current_data['filename'],
                        'suggested_label': current_data['suggested_label'],
                        'is_correct': True,
                        'corrected_label': '',
                        'annotator': username,
                        'status': 'correct',
                        'confidence_tag': confidence_tag,
                    })
                    
                    # Show success toast
                    st.success("✅ Saved as correct!", icon="✅")
                    
                    # Move to next
                    if st.session_state.current_index < len(filtered_data) - 1:
                        st.session_state.current_index += 1
                    st.rerun()
        
        with col_b:
            if is_correct == "incorrect":
                if st.button("💾 Submit & Next", type="primary", use_container_width=True, disabled=not corrected_label.strip()):
                    # Save annotation
                    storage.save_annotation({
                        'image_path': current_data['image_path'],
                        'folder': current_data['folder'],
                        'filename': current_data['filename'],
                        'suggested_label': current_data['suggested_label'],
                        'is_correct': False,
                        'corrected_label': corrected_label.strip(),
                        'annotator': username,
                        'status': 'corrected',
                        'confidence_tag': confidence_tag,
                    })
                    
                    # Show success toast
                    st.success(f"✅ Saved correction: '{corrected_label.strip()}'", icon="✅")
                    
                    # Move to next
                    if st.session_state.current_index < len(filtered_data) - 1:
                        st.session_state.current_index += 1
                    st.rerun()
        
        with col_c:
            if is_correct == "invalid":
                if st.button("⚠️ Mark Invalid & Next", type="secondary", use_container_width=True):
                    # Save annotation
                    storage.save_annotation({
                        'image_path': current_data['image_path'],
                        'folder': current_data['folder'],
                        'filename': current_data['filename'],
                        'suggested_label': current_data['suggested_label'],
                        'is_correct': False,
                        'corrected_label': 'INVALID_SAMPLE',
                        'annotator': username,
                        'status': 'invalid',
                        'confidence_tag': confidence_tag,
                    })
                    
                    # Show warning toast
                    st.warning("⚠️ Marked as invalid sample", icon="⚠️")
                    
                    # Move to next
                    if st.session_state.current_index < len(filtered_data) - 1:
                        st.session_state.current_index += 1
                    st.rerun()
    
    # Navigation
    st.markdown("---")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])
    
    with nav_col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0, use_container_width=True):
            st.session_state.current_index -= 1
            st.rerun()
    
    with nav_col2:
        if st.button("Next ➡️", disabled=st.session_state.current_index >= len(filtered_data) - 1, use_container_width=True):
            st.session_state.current_index += 1
            st.rerun()
    
    with nav_col3:
        jump_to = st.number_input(
            "Jump to image:",
            min_value=1,
            max_value=len(filtered_data),
            value=st.session_state.current_index + 1,
            key="jump_input"
        )
        if st.button("Go", use_container_width=True):
            st.session_state.current_index = jump_to - 1
            st.rerun()
