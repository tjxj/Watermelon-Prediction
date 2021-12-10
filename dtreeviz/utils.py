import xml.etree.cElementTree as ET
import pandas as pd
from numbers import Number
from typing import Tuple, Sequence


def inline_svg_images(svg) -> str:
    """
    Inline IMAGE tag refs in graphviz/dot -> SVG generated files.

    Convert all .svg image tag refs directly under g tags like:

    <g id="node1" class="node">
        <image xlink:href="/tmp/node4.svg" width="45px" height="76px" preserveAspectRatio="xMinYMin meet" x="76" y="-80"/>
    </g>

    to

    <g id="node1" class="node">
        <svg width="45px" height="76px" viewBox="0 0 49.008672 80.826687" preserveAspectRatio="xMinYMin meet" x="76" y="-80">
            XYZ
        </svg>
    </g>


    where XYZ is taken from ref'd svg image file:

    <?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
      "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <!-- Created with matplotlib (http://matplotlib.org/) -->
    <svg height="80.826687pt" version="1.1" viewBox="0 0 49.008672 80.826687" width="49.008672pt"
         xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        XYZ
    </svg>

    Note that width/height must be taken image ref tag and put onto svg tag. We
    also need the viewBox or it gets clipped a bit.

    :param svg: SVG string with <image/> tags.
    :return: svg with <image/> tags replaced with content of referenced svg image files.
    """
    ns = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.fromstring(svg)
    tree = ET.ElementTree(root)
    parent_map = {c: p for p in tree.iter() for c in p}

    # Find all image tags in document (must use svg namespace)
    image_tags = tree.findall(".//svg:g/svg:image", ns)
    for img in image_tags:
        # load ref'd image and get svg root
        svgfilename = img.attrib["{http://www.w3.org/1999/xlink}href"]
        with open(svgfilename, encoding='UTF-8') as f:
            imgsvg = f.read()
        imgroot = ET.fromstring(imgsvg)
        for k,v in img.attrib.items(): # copy IMAGE tag attributes to svg from image file
            if k not in {"{http://www.w3.org/1999/xlink}href"}:
                imgroot.attrib[k] = v
        # replace IMAGE with SVG tag
        p = parent_map[img]
        # print("BEFORE " + ', '.join([str(c) for c in p]))
        p.append(imgroot)
        p.remove(img)
        # print("AFTER " + ', '.join([str(c) for c in p]))

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    xml_str = ET.tostring(root).decode()
    return xml_str


def get_SVG_shape(svg) -> Tuple[Number,Number,Sequence[Number]]:
    """
    Sample line from SVG file from which we can get w,h,viewBox:
    <svg ... height="382pt" viewBox="0.00 0.00 344.00 382.00" width="344pt">
    Return:
    (344.0, 382.0, [0.0, 0.0, 344.0, 382.0])
    """
    root = ET.fromstring(svg)
    attrs = root.attrib
    viewBox = [float(v) for v in attrs['viewBox'].split(' ')]
    return (float(attrs['width'].strip('pt')),
            float(attrs['height'].strip('pt')),
            viewBox)


def scale_SVG(svg:str, scale:float) -> str:
    """
    Convert:

    <svg ... height="382pt" viewBox="0.00 0.00 344.00 382.00" width="344pt">
    <g class="graph" id="graph0" transform="scale(1 1) rotate(0) translate(4 378)">

    To:

    <svg ... height="191.0" viewBox="0.0 0.0 172.0 191.0" width="172.0">
    <g class="graph" id="graph0" transform="scale(.5 .5) rotate(0) translate(4 378)">
    """
    # Scale bounding box etc...
    w, h, viewBox = get_SVG_shape(svg)
    root = ET.fromstring(svg)
    root.set("width", str(w*scale))
    root.set("height", str(h*scale))
    viewBox[2] *= scale
    viewBox[3] *= scale
    root.set("viewBox", ' '.join([str(v) for v in viewBox]))

    # Deal with graph scale
    ns = {"svg": "http://www.w3.org/2000/svg"}
    graph = root.find(".//svg:g", ns) # get first node, which is graph
    transform = graph.attrib['transform']
    transform = transform.replace('scale(1 1)', f'scale({scale} {scale})')
    graph.set("transform", transform)

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    xml_str = ET.tostring(root).decode()
    return xml_str
    # print(root.attrib)
    # return ET.tostring(root, encoding='utf8', method='xml').decode("utf-8")

    # return root.tostring()#ET.tostring(root, 'utf-8')

def myround(v,ndigits=2):
    return format(v, '.' + str(ndigits) + 'f')


def _extract_final_feature_names(pipeline, features):
    """
    Computes the final features names of a :py:mod:`~sklearn.pipeline.Pipeline` used in its last
    component.

    Args:
        pipeline (sklearn.pipeline.Pipeline): A pipeline
        features (list): List of input features to the pipeline

    Returns:
        list: Features names used by the last component
    """
    for component in pipeline[:-1]:
        if hasattr(component, 'get_support'):
            features = [f for f, s in zip(features, component.get_support()) if s]
        if hasattr(component, 'get_feature_names'):
            features = component.get_feature_names(features)

    return features


def _normalize_class_names(class_names, nclasses):
    if class_names is None:
        return {i: f"class {i}" for i in range(nclasses)}
    if isinstance(class_names, dict):
        return class_names
    elif isinstance(class_names, Sequence):
        return {i: n for i, n in enumerate(class_names)}
    else:
        raise Exception(f"class_names must be dict or sequence, not {class_names.__class__.__name__}")


def extract_params_from_pipeline(pipeline, x_data, feature_names):
    """
    Extracts necessary parameters from an :py:class:`sklearn.pipeline.Pipeline` to pass into
    :py:class:`dtreeviz.models.sklearn_decision_trees.ShadowSKDTree`.

    Args:
        pipeline (sklearn.pipeline.Pipeline): An SKlearn pipeline whose last component is a decision tree model.
        x_data (numpy.ndarray): The (X)-input data on which the pipeline was fitted on.
        feature_names (list): List of names of the features in `x_data`.

    Returns:
        tuple: Tuple consisting of the tree model, the transformed input data, and a list of feature
        names used by the model.
    """

    # Pick last element of pipeline
    tree_model = pipeline.steps[-1][1]

    feature_names = _extract_final_feature_names(
        pipeline=pipeline,
        features=feature_names
    )
    x_data = pd.DataFrame(
        data=pipeline[:-1].transform(x_data),
        columns=feature_names
    )
    return tree_model, x_data, feature_names


def check_tree_index(tree_index, nr_of_trees):
    if tree_index is None:
        raise ValueError("You need to pass in a tree_index parameter.")
    if tree_index >= nr_of_trees:
        raise ValueError(f"tree_index parameter should have values between [{0}, {nr_of_trees - 1}].")


if __name__ == '__main__':
    # test rig
    with open("/tmp/t.svg") as f:
        svg = f.read()
        svg2 = scale_SVG(svg, scale=(.8))

    with open("/tmp/u.svg", "w") as f:
        f.write(svg2)
