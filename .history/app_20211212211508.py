import streamlit as st
from visualize import plotSurface,decisionTreeViz,svg_write
from DecisionTree import predictor,dt_param_selector

def md_contents():
    collapse_content = """
    <details>
    <summary>Getting started</summary>
    <a href="https://mp.weixin.qq.com/s/OFoTbEcvbe-k3Ys-wTStPA">streamlit 极简入门</a>
    </details>
    
    <details>
    <summary>Additional knowledge</summary>
    <a href="https://mp.weixin.qq.com/s/74ehblIzwe4rmCM6K-ynEA">决策树的可视化</a>
    
    </details>
    
    """
    st.title("决策树西瓜挑选器")
    st.markdown(collapse_content, unsafe_allow_html=True)

def body():    
    predictor()
    st.markdown("---")
    st.write("Source Code")
    with st.expander("data.py", expanded=False):
        with open("data.py", encoding="UTF-8") as f:
            st.code(f.read(), language="python")

    with st.expander("DecisionTree.py", expanded=False):
        with open("DecisionTree.py", encoding="UTF-8") as f:
            st.code(f.read(), language="python")

    with st.expander("visualize.py", expanded=False):
        with open("visualize.py", encoding="UTF-8") as f:
            st.code(f.read(), language="python")

    if st.checkbox("decisionTreeViz"):
        viz = decisionTreeViz(dt_param_selector())
        # svg = viz.svg()
        # svg_write(svg)
        st.write(viz)
        ps=plotSurface()

if __name__ == '__main__':
    md_contents()
    body()