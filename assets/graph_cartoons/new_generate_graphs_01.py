import sys
sys.path.append('../../../')
from causalign.config.paths import PathManager

# Initialize PathManager
paths = PathManager()

# graphs = [
#     {"filename": "graph_A", "C1": "inference", "C2": "observed", "E": "observed"},
#     {"filename": "graph_B", "C1": "inference", "C2": "unobserved", "E": "observed"},
#     {"filename": "graph_C", "C1": "inference", "C2": "unobserved", "E": "observed"},
#     {"filename": "graph_D", "C1": "inference", "C2": "observed", "E": "observed"},
#     {"filename": "graph_E", "C1": "inference", "C2": "unobserved", "E": "observed"},
#     {"filename": "graph_F", "C1": "inference", "C2": "observed", "E": "observed"},
#     {"filename": "graph_G", "C1": "inference", "C2": "observed", "E": "unobserved"},
#     {"filename": "graph_H", "C1": "inference", "C2": "observed", "E": "unobserved"},
#     {"filename": "graph_I", "C1": "observed", "C2": "observed", "E": "inference"},
#     {"filename": "graph_J", "C1": "observed", "C2": "observed", "E": "inference"},
#     {"filename": "graph_K", "C1": "observed", "C2": "observed", "E": "inference"},
# ]

# template = r"""
# \documentclass[tikz,border=2mm]{{standalone}}

# \usetikzlibrary{{arrows.meta,positioning}}
# \usepackage{{xcolor}}
# \begin{{document}}
# \definecolor{{tuebingenred}}{{RGB}}{{126,18,24}}
# \definecolor{{tuebingengray}}{{RGB}}{{88,89,91}}
# \definecolor{{darkgray}}{{RGB}}{{46,46,46}}

# \tikzset{{
#     observed/.style={{circle, draw, fill=darkgray!90, minimum size=9mm, text width=9mm, text=white, align=center}},
#     unobserved/.style={{circle, draw, fill=white, minimum size=9mm, text width=9mm, align=center}},
#     inference/.style={{circle, draw, fill=tuebingenred, minimum size=9mm, text width=9mm, text=white, align=center}},
#     arrow/.style={{-{{Latex}}[length=2mm], thick}},
# }}


# \begin{{tikzpicture}}
# \node[{{C1}}] (C1) at (0, 1.5) {{$C_i :1$ }};
# \node[{{C2}}] (C2) at (2, 1.5) {{$C_j :1$}};
# \node[{{E}}] (E) at (1, 0) {{$E:1$}};
# \draw[arrow] (C1) -- (E);
# \draw[arrow] (C2) -- (E);
# \end{{tikzpicture}}
# \end{{document}}
# """

# # Generate a .tex file for each graph
# for graph in graphs:
#     print(graph)
#     content = template.format(C1=graph["C1"], C2=graph["C2"], E=graph["E"])
#     print(content)
#     with open(f"{graph['filename']}.tex", "w") as f:
#         f.write(content)


# graphs = [
#     {"filename": "graph_A", "C1": "inference", "C2": "observed", "E": "observed"},
#     {"filename": "graph_B", "C1": "inference", "C2": "unobserved", "E": "observed"},
#     {"filename": "graph_C", "C1": "inference", "C2": "unobserved", "E": "observed"},
#     {"filename": "graph_D", "C1": "inference", "C2": "observed", "E": "observed"},
#     {"filename": "graph_E", "C1": "inference", "C2": "unobserved", "E": "observed"},
#     {"filename": "graph_F", "C1": "inference", "C2": "observed", "E": "observed"},
#     {"filename": "graph_G", "C1": "inference", "C2": "observed", "E": "unobserved"},
#     {"filename": "graph_H", "C1": "inference", "C2": "observed", "E": "unobserved"},
#     {"filename": "graph_I", "C1": "observed", "C2": "observed", "E": "inference"},
#     {"filename": "graph_J", "C1": "observed", "C2": "observed", "E": "inference"},
#     {"filename": "graph_K", "C1": "observed", "C2": "observed", "E": "inference"},
# ]

# graphs = [
#     {
#         "filename": "graph_A",
#         "C1": "inference",
#         "C2": "observed",
#         "E": "observed",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :1",
#         "E_text": "E:1",
#     },
#     {
#         "filename": "graph_B",
#         "C1": "inference",
#         "C2": "unobserved",
#         "E": "observed",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j",
#         "E_text": "E:1",
#     },
#     {
#         "filename": "graph_C",
#         "C1": "inference",
#         "C2": "observed",
#         "E": "observed",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :0",
#         "E_text": "E:1",
#     },
#     {
#         "filename": "graph_D",
#         "C1": "inference",
#         "C2": "observed",
#         "E": "observed",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :1",
#         "E_text": "E:0",
#     },
#     {
#         "filename": "graph_E",
#         "C1": "inference",
#         "C2": "unobserved",
#         "E": "observed",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j",
#         "E_text": "E:0",
#     },
#     {
#         "filename": "graph_F",
#         "C1": "inference",
#         "C2": "observed",
#         "E": "observed",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :0",
#         "E_text": "E:0",
#     },
#     {
#         "filename": "graph_G",
#         "C1": "inference",
#         "C2": "observed",
#         "E": "unobserved",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :1",
#         "E_text": "E",
#     },
#     {
#         "filename": "graph_H",
#         "C1": "inference",
#         "C2": "observed",
#         "E": "unobserved",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :0",
#         "E_text": "E",
#     },
#     {
#         "filename": "graph_I",
#         "C1": "observed",
#         "C2": "observed",
#         "E": "inference",
#         "C1_text": "C_i :0",
#         "C2_text": "C_j :0",
#         "E_text": "E :1",
#     },
#     {
#         "filename": "graph_J",
#         "C1": "observed",
#         "C2": "observed",
#         "E": "inference",
#         "C1_text": "C_i :0",
#         "C2_text": "C_j :1",
#         "E_text": "E :1",
#     },
#     {
#         "filename": "graph_K",
#         "C1": "observed",
#         "C2": "observed",
#         "E": "inference",
#         "C1_text": "C_i :1",
#         "C2_text": "C_j :1",
#         "E_text": "E :1",
#     },
# ]


# second label change
# acutally, third label change, based on second label change
# where labels are sorted s.t. human data is always monotonicly increasing
#  this means that D and E are swapped and J and K are swapped


# labeled after original labels in RW-17 raw human data (collider, probably also true for fork)
graphs = [
    # effect present diag inference
    {
        "filename": "01_graph_a",
        "C1": "inference",
        "C2": "observed",
        "E": "observed",
        "C1_text": "1",
        "C2_text": "1",
        "E_text": "1",
    },
    {
        "filename": "01_graph_b",
        "C1": "inference",
        "C2": "unobserved",
        "E": "observed",
        "C1_text": "1",
        "C2_text": "",
        "E_text": "1",
    },
    {
        "filename": "01_graph_c",
        "C1": "inference",
        "C2": "observed",
        "E": "observed",
        "C1_text": "1",
        "C2_text": "0",
        "E_text": "1",
    },
    # absent based diag
    {
        "filename": "01_graph_f",
        "C1": "inference",
        "C2": "observed",
        "E": "observed",
        "C1_text": "1",
        "C2_text": "1",
        "E_text": "0",
    },
    {
        "filename": "01_graph_g",
        "C1": "inference",
        "C2": "unobserved",
        "E": "observed",
        "C1_text": "1",
        "C2_text": "",
        "E_text": "0",
    },
    {
        "filename": "01_graph_h",
        "C1": "inference",
        "C2": "observed",
        "E": "observed",
        "C1_text": "1",
        "C2_text": "0",
        "E_text": "0",
    },
    # conditional independence inference
    {
        "filename": "01_graph_d",
        "C1": "inference",
        "C2": "observed",
        "E": "unobserved",
        "C1_text": "1",
        "C2_text": "1",
        "E_text": "",
    },
    {
        "filename": "01_graph_e",
        "C1": "inference",
        "C2": "observed",
        "E": "unobserved",
        "C1_text": "1",
        "C2_text": "0",
        "E_text": "",
    },
    # predictive inference
    {
        "filename": "01_graph_i",
        "C1": "observed",
        "C2": "observed",
        "E": "inference",
        "C1_text": "0",
        "C2_text": "0",
        "E_text": "1",
    },
    {
        "filename": "01_graph_j",
        "C1": "observed",
        "C2": "observed",
        "E": "inference",
        "C1_text": "0",
        "C2_text": "1",
        "E_text": "1",
    },
    {
        "filename": "01_graph_k",
        "C1": "observed",
        "C2": "observed",
        "E": "inference",
        "C1_text": "1",
        "C2_text": "1",
        "E_text": "1",
    },
]


# template = r"""
# \documentclass[tikz,border=2mm]{{standalone}}
# \usetikzlibrary{{arrows.meta,positioning}}
# \usepackage{{xcolor}}
# \begin{{document}}

# \definecolor{{tuebingenred}}{{RGB}}{{126,18,24}}
# \definecolor{{tuebingengray}}{{RGB}}{{88,89,91}}
# \definecolor{{darkgray}}{{RGB}}{{46,46,46}}

# \tikzset{{
#     observed/.style={{circle, draw, fill=darkgray!90, minimum size=9mm, text width=9mm, text=white, align=center}},
#     unobserved/.style={{circle, draw, fill=white, minimum size=9mm, text width=9mm, align=center}},
#     inference/.style={{circle, draw, fill=tuebingenred, minimum size=9mm, text width=9mm, text=white, align=center}},
#     arrow/.style={{-{{Latex[length=2mm]}}, thick}},
# }}


# \begin{{tikzpicture}}
# \node[{C1}] (C1) at (0, 1.5) {{$C_i :1$}};
# \node[{C2}] (C2) at (2, 1.5) {{$C_j :1$}};
# \node[{E}] (E) at (1, 0) {{$E:1$}};
# \draw[arrow] (C1) -- (E);
# \draw[arrow] (C2) -- (E);
# \end{{tikzpicture}}
# \end{{document}}
# """


# \begin{{tikzpicture}}
# \node[{C1}] (C1) at (0, 1.5) {{${C1_text}$}};
# \node[{C2}] (C2) at (2, 1.5) {{${C2_text}$}};
# \node[{E}] (E) at (1, 0) {{${E_text}$}};
# \draw[arrow] (C1) -- (E);
# \draw[arrow] (C2) -- (E);
# \end{{tikzpicture}}
# \end{{document}}
# \end{{document}}
# """


template = r"""
\documentclass[tikz,border=2mm]{{standalone}}
\usetikzlibrary{{arrows.meta,positioning}}
\usepackage{{xcolor}}
\begin{{document}}
\definecolor{{tuebingenred}}{{RGB}}{{126,18,24}}
\definecolor{{tuebingengray}}{{RGB}}{{88,89,91}}
\definecolor{{darkgray}}{{RGB}}{{46,46,46}}
\definecolor{{observed}}{{HTML}}{{AAAAAA}} %C6C6C6
\definecolor{{inference}}{{HTML}}{{FF5B59}} % FF5B59 FFB9B8

\tikzset{{
    observed/.style={{circle, draw, fill=observed, minimum size=11mm, text width=11mm, text=black, align=center}},
    unobserved/.style={{circle, draw, fill=white, minimum size=11mm, text width=11mm, align=center, dashed, very thick}},
    inference/.style={{circle, draw, fill=inference, minimum size=11mm, text width=11mm, text=black, align=center}},
    arrow/.style={{-{{Latex[length=4mm, width=4mm]}}, line width=1.2mm}},
}}




\begin{{tikzpicture}}
\node[{C1}] (C1) at (0, 2.2) {{\Huge${{\mathbf{{{C1_text}}}}}$}};
\node[{C2}] (C2) at (2, 2.2) {{\Huge${{\mathbf{{{C2_text}}}}}$}};
\node[{E}] (E) at (1, 0.2) {{\Huge${{\mathbf{{{E_text}}}}}$}};  
\draw[arrow] (C1) -- (E);
\draw[arrow] (C2) -- (E);
\end{{tikzpicture}}
\end{{document}}"""
# # Generate a .tex file for each graph
# for graph in graphs:
#     content = template.format(C1=graph["C1"], C2=graph["C2"], E=graph["E"])
#     with open(f"{graph['filename']}.tex", "w") as f:
#         f.write(content)
# Generate a .tex file for each graph
for graph in graphs:
    content = template.format(
        C1=graph["C1"],
        C2=graph["C2"],
        E=graph["E"],
        C1_text=graph["C1_text"],
        C2_text=graph["C2_text"],
        E_text=graph["E_text"],
    )
    with open(f"{graph['filename']}.tex", "w") as f:
        f.write(content)
