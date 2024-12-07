import streamlit as st
from main import IR_Ret_Sys
from llama_tokenise_rag import summarize_documents_with_query

IR = IR_Ret_Sys()

def generate_ai_summary(query, abstracts):
    return summarize_documents_with_query(query, abstracts, IR.text_gen_pipeline)


def get_search_results(query):
    #### Check which Query function to use! ####
    docid_score = IR.TFIDFRanker.query_augmented(query)
    results = [
        {
        'title': IR.docid_title_map.get(i[0]),
        'link' : IR.docid_link_map.get(i[0]),
        'snippet': IR.docid_abstract_map.get(i[0])
        }
        for i in docid_score[:100]
    ]
    return results

if "query_submitted" not in st.session_state:
    st.session_state["query_submitted"] = False

if "query" not in st.session_state:
    st.session_state["query"] = ""

if "previous_query" not in st.session_state:
    st.session_state["previous_query"] = None

if "ai_summary" not in st.session_state:
    st.session_state["ai_summary"] = None

if "show_full_summary" not in st.session_state:
    st.session_state["show_full_summary"] = False

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

    # Check if the query has changed
    if query != st.session_state["previous_query"]:
        st.session_state["ai_summary"] = None  # Reset cached summary
        st.session_state["show_full_summary"] = False  # Reset toggle
        st.session_state["previous_query"] = query  # Update stored query

    if query:
        # Create a placeholder for the summary **at the top**
        summary_placeholder = st.empty()

        # Render "Generating summary" dynamically
        if st.session_state["ai_summary"] is None:
            summary_placeholder.markdown(
                """
                <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #FFC107; border-radius: 5px; margin-bottom: 0px;">
                    <h4 style="color: #333;">AI-Generated Overview</h4>
                    <p style="font-size: 16px; color: #555;">Generating summary...</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Fetch search results
        with st.spinner("Fetching results..."):
            search_results = get_search_results(query)

        # Display search results immediately
        st.subheader("Top 20 Results")
        for i, result in enumerate(search_results[:20]):  # Limit display to top 20
            st.markdown(f"**{i+1}. [{result['title']}]({result['link']})**")
            st.write(result['snippet'])
            st.markdown("---")

        # Generate the summary in the background if not cached
        if st.session_state["ai_summary"] is None:
            with st.spinner("Generating summary..."):
                st.session_state["ai_summary"] = generate_ai_summary(query, search_results)

        # Retrieve the cached summary
        ai_summary = st.session_state["ai_summary"]
        preview_length = 200
        summary_preview = ai_summary[:preview_length] + "..." if len(ai_summary) > preview_length else ai_summary

        # Display the appropriate summary
        displayed_summary = ai_summary if st.session_state["show_full_summary"] else summary_preview
        with summary_placeholder.container():
            st.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #4CAF50; border-radius: 5px; margin-bottom: 0px;">
                    <h4 style="color: #333;">AI-Generated Overview</h4>
                    <p style="font-size: 16px; color: #555;">{displayed_summary}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Add toggle button near the summary
            button_label = "Show More" if not st.session_state["show_full_summary"] else "Show Less"
            if st.button(button_label, key="summary_toggle"):
                st.session_state["show_full_summary"] = not st.session_state["show_full_summary"]
