if not all_candidates:
    st.info(
        "No matches were found in the vector DB based on this description."
    )
else:
    st.markdown('<div class="section-title">📌 Candidate Matches</div>', unsafe_allow_html=True)

    if not filtered:
        st.info(
            "No matches under the current distance threshold. "
            "Showing raw top-K candidates instead."
        )
        to_show = all_candidates
    else:
        to_show = filtered

    for doc, score in to_show:
        meta = doc.metadata or {}
        similarity_pct = max(0.0, (1.0 - score) * 100.0)

        st.markdown(
            f"**Distance:** `{score:.4f}`  ·  "
            f"**Similarity (approx):** `{similarity_pct:.1f}%`"
        )

        st.write("**Description:**", meta.get("description", doc.page_content))

        if meta.get("subway_location"):
            st.write("🚉 Location:", meta["subway_location"])
        if meta.get("color"):
            st.write("🎨 Color:", meta["color"])
        if meta.get("item_category"):
            st.write("📂 Category:", meta["item_category"])
        if meta.get("item_type"):
            st.write("🔖 Type:", meta["item_type"])

        st.caption(
            f"Found item ID: {meta.get('found_id', 'N/A')} · Time: {meta.get('time', '')}"
        )
        with st.expander("View raw metadata"):
            st.json(meta)
        st.markdown("---")

