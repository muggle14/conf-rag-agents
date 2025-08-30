# src/confluence_qa/utils/page_tree.py   (update)


def build_tree(hit_docs, highlight_ids):
    """
    hit_docs  : iterable of Azure Search docs (must include 'id','title','url')
    highlight_ids : set[str] of doc.id to mark with ⭐
    Returns a Markdown tree with clickable links.
    """
    by_id = {d["id"]: d for d in hit_docs}

    # ---- find roots ----
    roots = set()
    for d in by_id.values():
        p = d.get("parent_page_id")
        if not p or p not in by_id:
            roots.add(d["id"])

    # ---- DFS ----
    def walk(node_id, depth=0):
        d = by_id[node_id]
        star = " ⭐" if node_id in highlight_ids else ""
        indent = "  " * depth
        line = f"{indent}- [{d['title']}]({d['url']}){star}\n"
        for c in d.get("child_ids", []):
            if c in by_id:
                line += walk(c, depth + 1)
        return line

    return "\n".join(walk(r) for r in roots)
