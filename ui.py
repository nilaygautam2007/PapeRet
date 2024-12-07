import streamlit as st


def generate_ai_summary(query):
    return (
        f"This is an AI-generated overview for the query: '{query}'. It provides a concise summary of the most relevant information. "
        f"This overview includes a lot of details about the topic, helping users quickly grasp the most important aspects. "
        f"Here we discuss multiple facets of the query in depth, providing insights and additional context to make the information "
        f"more comprehensive and valuable."
    )


def get_search_results(query):
    results = [
        {"title": f"Result {i+1} Title", "link": f"https://example.com/result{i+1}", "snippet": f"Snippet for result {i+1}"}
        for i in range(20)
    ]
    return results

if "query_submitted" not in st.session_state:
    st.session_state["query_submitted"] = False

if "query" not in st.session_state:
    st.session_state["query"] = ""

st.markdown(
    """
    <style>
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 20vh;
        flex-direction: column;
    }
    .query-box {
        width: 60%;
        max-width: 500px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["query_submitted"]:
    # Initial page: Query input only
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; '>LLAMA Paper Search</h1>", unsafe_allow_html=True)
    query = st.text_input("", placeholder="Type your query here...")
    col1, col2, col3 , col4, col5 = st.columns(5)

    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        if st.button("Search"):
            if query.strip():  # Ensure the query is not empty
                st.session_state["query"] = query
                st.session_state["query_submitted"] = True
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.title("LLAMA Paper Search")
    st.write("Enter your query below.")
    query = st.session_state["query"]
    query = st.text_input("Search Query:", placeholder="Type your query here...", value=query)

    if query:
        with st.spinner("Generating summary..."):
            ai_summary = generate_ai_summary(query)
        
        preview_length = 200
        if len(ai_summary) > preview_length:
            summary_preview = ai_summary[:preview_length] + "..."
        else:
            summary_preview = ai_summary

        # Initialize session state for toggling
        if "show_full_summary" not in st.session_state:
            st.session_state["show_full_summary"] = False

        # Display the appropriate summary
        displayed_summary = ai_summary if st.session_state["show_full_summary"] else summary_preview
        st.markdown(
            f"""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #4CAF50; border-radius: 5px; margin-bottom: 0px;">
                <h4 style="color: #333;">AI-Generated Overview</h4>
                <p style="font-size: 16px; color: #555;">{displayed_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        

        col1, col2, col3 , col4, col5 = st.columns(5)

        with col1:
            pass
        with col2:
            pass
        with col4:
            pass
        with col5:
            pass
        with col3 :
            # center_button = st.button('Button')
            button_label = "Show More" if not st.session_state["show_full_summary"] else "Show Less"

            if st.button(button_label):
                st.session_state["show_full_summary"] = not st.session_state["show_full_summary"]


        with st.spinner("Fetching results..."):
            search_results = get_search_results(query)

        st.subheader("Top 20 Results")
        for i, result in enumerate(search_results):
            st.markdown(f"**{i+1}. [{result['title']}]({result['link']})**")
            st.write(result['snippet'])
            st.markdown("---")
