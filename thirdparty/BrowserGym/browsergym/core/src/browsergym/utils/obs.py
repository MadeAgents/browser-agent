# =============================================================================
# MODIFICATIONS:
# This file has been modified from the original BrowserGym open-source project
# Modification Date: 2025-12-22
# Main Changes:
#   - Added overlay_som function to support Set-of-Marks visualization
#   - Observation processing and rendering logic may have been enhanced
# =============================================================================

import ast
import logging
import math
import re
from collections import defaultdict

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from bs4 import BeautifulSoup

from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE as BID_ATTR
from browsergym.core.constants import BROWSERGYM_SETOFMARKS_ATTRIBUTE as SOM_ATTR
from browsergym.core.constants import BROWSERGYM_VISIBILITY_ATTRIBUTE as VIS_ATTR

logger = logging.getLogger(__name__)

IGNORED_AXTREE_ROLES = ["LineBreak"]

IGNORED_AXTREE_PROPERTIES = (
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "focusable",
)

RETAINED_PROPERTIES = ["required", "disabled", "checked", "valuemin", "valuemax", "valuetext", "selected", "page_dialog_message"]
UNWANTED_PROPERTIES = ["focused", "autocomplete", "hasPopup", "expanded", "multiselectable", "orientation", "controls"]
UNINTERACTIVE_ROLES = ["StaticText", "LabelText", "main", "heading", "LayoutTable", "tabpanel", "LayoutTableRow", "LayoutTableCell", "time", "list", "contentinfo", "table", "row", "rowheader", "columnheader", "gridcell", "caption", "DescriptionList", "DescriptionListTerm", "DescriptionListDetail", "RootWebArea", "rowgroup", "alert"]
ROLE_REPLACEMENT_DICT = {
    "StaticText": "text",
    "LabelText": "text",
    # "caption": "text",
    # "generic": "text"
}

def flatten_dom_to_str(
    dom_snapshot,
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    hide_bid_if_invisible: int = False,
    hide_all_bids: bool = False,
) -> str:
    """Formats a DOM snapshot into a string text"""

    def to_string(idx):
        if idx == -1:
            return None
        else:
            return dom_snapshot["strings"][idx]

    def parse_document(document_idx) -> str:
        # adapted from [natbot](https://github.com/nat/natbot)

        nodes = dom_snapshot["documents"][document_idx]["nodes"]
        node_children = defaultdict(lambda: [])

        for node_idx in range(len(nodes["nodeName"])):
            parent_idx = nodes["parentIndex"][node_idx]
            if parent_idx != -1:
                node_children[parent_idx].append(node_idx)

        def dfs(node_idx: int, parent_node_skipped: bool) -> str:

            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeType
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeName
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeValue

            node_type = nodes["nodeType"][node_idx]
            node_name = to_string(nodes["nodeName"][node_idx])
            node_value = to_string(nodes["nodeValue"][node_idx])
            html_before = ""
            html_after = ""
            skip_node = False

            # text nodes: print text content only if parent was not skipped
            if node_type == 3:  # node_name == "#text"
                if not parent_node_skipped and node_value is not None:
                    html_before += node_value

            # CData nodes: print content only if parent was not skipped
            elif node_type == 4:  # node_name == "#cdata-section":
                if not parent_node_skipped and node_value is not None:
                    html_before += f"<!CDATA[[{node_value}]]>"

            # processing instructions, comments, documents, doctypes, document fragments: don't print
            elif node_type in (7, 8, 9, 10, 11):
                skip_node = True

            # now we should have an element node
            else:
                assert node_type == 1

                tag_name = node_name.lower().strip()
                attributes = []  # to be printed as attributes with the tag
                bid = None

                # parse node attributes
                node_attr_idxs = nodes["attributes"][node_idx]
                for i in range(0, len(node_attr_idxs), 2):
                    attr_name = to_string(node_attr_idxs[i])
                    attr_value = to_string(node_attr_idxs[i + 1])

                    # extract and print bid
                    if attr_name == BID_ATTR:
                        bid = attr_value
                    # ignore browsergym attributes
                    elif attr_name in (VIS_ATTR, SOM_ATTR):
                        pass
                    # print other attributes
                    else:
                        if attr_value is None:
                            # attribute value missing
                            attributes.append(f"{attr_name}")
                        else:
                            # attribute value present
                            attributes.append(f'{attr_name}="{attr_value}"')

                skip_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=extra_properties,
                    with_visible=with_visible,
                    with_clickable=with_clickable,
                    with_center_coords=with_center_coords,
                    with_bounding_box_coords=with_bounding_box_coords,
                    with_som=with_som,
                    filter_visible_only=filter_visible_only,
                    filter_with_bid_only=filter_with_bid_only,
                    filter_som_only=filter_som_only,
                    coord_decimals=coord_decimals,
                )

                # insert extra attributes before regular attributes
                attributes = extra_attributes_to_print + attributes

                # insert bid as first attribute
                if not (
                    hide_all_bids
                    or bid is None
                    or (
                        hide_bid_if_invisible
                        and extra_properties.get(bid, {}).get("visibility", 0) < 0.5
                    )
                ):
                    attributes.insert(0, f'bid="{bid}"')

                if not skip_node:
                    # print node opening tag, with its attributes
                    html_before += f"<{tag_name}" + " ".join([""] + attributes) + ">"
                    # print node closing tag
                    html_after += f"</{tag_name}>"

            html = ""
            html += html_before

            # recursively print iframe nodes if any
            if node_idx in nodes["contentDocumentIndex"]["index"]:
                sub_document_idx = nodes["contentDocumentIndex"]["value"][
                    nodes["contentDocumentIndex"]["index"].index(node_idx)
                ]
                html += parse_document(document_idx=sub_document_idx)

            # recursively print children nodes if any
            for child_idx in node_children[node_idx]:
                html += dfs(node_idx=child_idx, parent_node_skipped=skip_node)

            html += html_after

            return html

        html = dfs(node_idx=0, parent_node_skipped=False)

        # Format the HTML document with indentation
        soup = BeautifulSoup(html, "lxml")
        html = soup.prettify()

        return html

    html = parse_document(document_idx=0)

    return html


def _get_coord_str(coord, decimals):
    if isinstance(coord, str):
        coord = list(map(float, ast.literal_eval(coord)))

    coord_format = f".{decimals}f"
    coord_str = ",".join([f"{c:{coord_format}}" for c in coord])
    return f"({coord_str})"


def _process_bid(
    bid,
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
):
    """
    Process extra attributes and attribute-based filters, for the element with the given bid.

    Returns:
        A flag indicating if the element should be skipped or not (due to filters).
        Attributes to be printed, as a list of "x=y" strings.
    """

    if extra_properties is None:
        if any(
            (
                with_visible,
                with_clickable,
                with_center_coords,
                with_bounding_box_coords,
                with_som,
                filter_visible_only,
                filter_with_bid_only,
                filter_som_only,
            )
        ):
            raise ValueError("extra_properties argument required")
        else:
            extra_properties = {}

    skip_element = False
    attributes_to_print = []

    if bid is None:
        # skip nodes without a bid (if requested)
        if filter_with_bid_only:
            skip_element = True
        if filter_som_only:
            skip_element = True
        if filter_visible_only:
            # element without bid have no visibility mark, they could be visible or non-visible
            # TODO we consider them as visible. Is this what we want? Now that duplicate bids are handled, should we mark all non-html elements?
            pass  # keep elements without visible property
            # skip_element = True  # filter elements without visible property

    # parse extra browsergym properties, if node has a bid
    else:
        if bid in extra_properties:
            node_vis = extra_properties[bid]["visibility"]
            node_bbox = extra_properties[bid]["bbox"]
            node_is_clickable = extra_properties[bid]["clickable"]
            node_in_som = extra_properties[bid]["set_of_marks"]
            node_is_visible = node_vis >= 0.5
            # skip non-visible nodes (if requested)
            if filter_visible_only and not node_is_visible:
                skip_element = True
            if filter_som_only and not node_in_som:
                skip_element = True
            # print extra attributes if requested (with new names)
            if with_som and node_in_som:
                attributes_to_print.insert(0, f"som")
            if with_visible and node_is_visible:
                attributes_to_print.insert(0, f"visible")
            if with_clickable and node_is_clickable:
                attributes_to_print.insert(0, f"clickable")
            if with_center_coords and node_bbox is not None:
                x, y, width, height = node_bbox
                center = (x + width / 2, y + height / 2)
                attributes_to_print.insert(0, f'center="{_get_coord_str(center, coord_decimals)}"')
            if with_bounding_box_coords and node_bbox is not None:
                x, y, width, height = node_bbox
                box = (x, y, x + width, y + height)
                attributes_to_print.insert(0, f'box="{_get_coord_str(box, coord_decimals)}"')

    return skip_element, attributes_to_print


def flatten_axtree_to_str(
    AX_tree,
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_roles=IGNORED_AXTREE_ROLES,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    remove_redundant_static_text: bool = True,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    hide_all_bids: bool = False,
) -> str:
    """Formats the accessibility tree into a string text"""
    node_id_to_idx = {}
    for idx, node in enumerate(AX_tree["nodes"]):
        node_id_to_idx[node["nodeId"]] = idx

    def dfs(node_idx: int, depth: int, parent_node_filtered: bool, parent_node_name: str) -> str:
        tree_str = ""
        node = AX_tree["nodes"][node_idx]
        indent = "\t" * depth
        skip_node = False  # node will not be printed, with no effect on children nodes
        filter_node = False  # node will not be printed, possibly along with its children nodes
        node_role = node["role"]["value"]
        node_name = ""

        if node_role in ignored_roles:
            skip_node = True
            pass
        elif "name" not in node:
            skip_node = True
            pass
        else:
            node_name = node["name"]["value"]
            if "value" in node and "value" in node["value"]:
                node_value = node["value"]["value"]
            else:
                node_value = None

            # extract bid
            bid = node.get("browsergym_id", None)

            # extract node attributes
            attributes = []
            for property in node.get("properties", []):
                if not "value" in property:
                    continue
                if not "value" in property["value"]:
                    continue

                prop_name = property["name"]
                prop_value = property["value"]["value"]

                if prop_name in ignored_properties:
                    continue
                elif prop_name in ("required", "focused", "atomic"):
                    if prop_value:
                        attributes.append(prop_name)
                else:
                    attributes.append(f"{prop_name}={repr(prop_value)}")

            if skip_generic and node_role == "generic" and not attributes:
                skip_node = True

            if hide_all_children and parent_node_filtered:
                skip_node = True

            if node_role == "StaticText":
                if parent_node_filtered:
                    skip_node = True
                elif remove_redundant_static_text and node_name in parent_node_name:
                    skip_node = True
            else:
                filter_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=extra_properties,
                    with_visible=with_visible,
                    with_clickable=with_clickable,
                    with_center_coords=with_center_coords,
                    with_bounding_box_coords=with_bounding_box_coords,
                    with_som=with_som,
                    filter_visible_only=filter_visible_only,
                    filter_with_bid_only=filter_with_bid_only,
                    filter_som_only=filter_som_only,
                    coord_decimals=coord_decimals,
                )

                # if either is True, skip the node
                skip_node = skip_node or filter_node

                # insert extra attributes before regular attributes
                attributes = extra_attributes_to_print + attributes

            # actually print the node string
            if not skip_node:
                if node_role == "generic" and not node_name:
                    node_str = f"{node_role}"
                else:
                    node_str = f"{node_role} {repr(node_name.strip())}"

                if not (
                    hide_all_bids
                    or bid is None
                    or (
                        hide_bid_if_invisible
                        and extra_properties.get(bid, {}).get("visibility", 0) < 0.5
                    )
                ):
                    node_str = f"[{bid}] " + node_str

                if node_value is not None:
                    node_str += f' value={repr(node["value"]["value"])}'

                if attributes:
                    node_str += ", ".join([""] + attributes)

                tree_str += f"{indent}{node_str}"

        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            # mark this to save some tokens
            child_depth = depth if skip_node else (depth + 1)
            child_str = dfs(
                node_id_to_idx[child_node_id],
                child_depth,
                parent_node_filtered=filter_node,
                parent_node_name=node_name,
            )
            if child_str:
                if tree_str:
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    tree_str = dfs(0, 0, False, "")
    return tree_str


def overlay_som(
    screenshot: np.typing.ArrayLike,
    extra_properties: dict,
    fontsize: int = 12,
    linewidth: int = 2,
    tag_margin: int = 2,
):
    img = PIL.Image.fromarray(screenshot).copy()  # make a copy
    img = img.convert(mode="RGBA")
    draw = PIL.ImageDraw.Draw(img)

    font = PIL.ImageFont.load_default(size=fontsize)

    # Adapted from https://stackoverflow.com/questions/51908563/dotted-or-dashed-line-with-python-pillow/58885306#58885306
    def linedashed(
        draw: PIL.ImageDraw.Draw, x0, y0, x1, y1, fill, width, dash_length=4, nodash_length=8
    ):
        line_dx = x1 - x0  # delta x (can be negative)
        line_dy = y1 - y0  # delta y (can be negative)
        line_length = math.hypot(line_dx, line_dy)  # line length (positive)
        if line_length == 0:
            return  # Avoid division by zero in case the line length is 0
        pixel_dx = line_dx / line_length  # x add for 1px line length
        pixel_dy = line_dy / line_length  # y add for 1px line length
        dash_start = 0
        while dash_start < line_length:
            dash_end = dash_start + dash_length
            if dash_end > line_length:
                dash_end = line_length
            draw.line(
                (
                    round(x0 + pixel_dx * dash_start),
                    round(y0 + pixel_dy * dash_start),
                    round(x0 + pixel_dx * dash_end),
                    round(y0 + pixel_dy * dash_end),
                ),
                fill=fill,
                width=width,
            )
            dash_start += dash_length + nodash_length

    for bid, properties in extra_properties.items():
        if properties["set_of_marks"] and properties["bbox"]:
            x, y, width, height = properties["bbox"]
            x0, y0 = x, y
            x1, y1 = x + width, y + height

            # skip small boxes
            area = (x1 - x0) * (y1 - y0)
            if area < 20:
                logger.warning(
                    f'som overlay: skipping bid "{bid}" due to bbox too small (area={area})'
                )
                continue

            # draw bounding box with dashed lines
            linedashed(draw, x0, y0, x1, y0, fill=(0, 0, 0, 255), width=linewidth)
            linedashed(draw, x1, y0, x1, y1, fill=(0, 0, 0, 255), width=linewidth)
            linedashed(draw, x1, y1, x0, y1, fill=(0, 0, 0, 255), width=linewidth)
            linedashed(draw, x0, y1, x0, y0, fill=(0, 0, 0, 255), width=linewidth)

            # get text box size (left, top, right, bottom)
            tag_box = font.getbbox(
                bid,
            )

            # set tag size, including margins
            tag_size = (
                (tag_box[2] - tag_box[0] + 2 * (tag_margin + 1)),
                (tag_box[3] - tag_box[1] + 2 * (tag_margin + 1)),
            )

            # create tag image with correct size and black background
            tag_img = PIL.Image.new("RGBA", tag_size, "black")
            tag_draw = PIL.ImageDraw.Draw(tag_img)
            # write text with 1px horizontal margin
            tag_draw.text(
                (-tag_box[0] + tag_margin + 1, -tag_box[1] + tag_margin + 1),
                bid,
                font=font,
                fill=(255, 255, 255, 255),
                spacing=0,
            )
            tag_draw.rectangle(
                (0, 0, tag_size[0] - 1, tag_size[1] - 1),
                fill=None,
                outline=(255, 255, 255, 255),
                width=1,
            )

            # draw tag in the source image, upper left of the bounding box
            tag_pos = (x + 0, y - tag_size[1] / 2 + 4)
            tag_pos = list(map(round, tag_pos))
            img.paste(tag_img, tag_pos)

    # convert to RGB (3 channels)
    img = img.convert(mode="RGB")
    # convert to a numpy array
    img = np.array(img)

    return img


def prune_html(html):
    html = re.sub(r"\n", " ", html)
    # remove html comments
    html = re.sub(r"<!--(.*?)-->", "", html, flags=re.MULTILINE)

    soup = BeautifulSoup(html, "lxml")
    for tag in reversed(soup.find_all()):
        # remove body and html tags (not their content)
        if tag.name in ("html", "body"):
            tag.unwrap()
        # remove useless tags
        elif tag.name in ("style", "link", "script", "br"):
            tag.decompose()
        # remove / unwrap structural tags
        elif tag.name in ("div", "span", "i", "p") and len(tag.attrs) == 1 and tag.has_attr("bid"):
            if not tag.contents:
                tag.decompose()
            else:
                tag.unwrap()

    html = soup.prettify()

    return html


def flatten_axtree_to_str_new(AX_tree, extra_properties: dict = None) -> str:
    """
    New function to flatten the accessibility tree into a string with AgentOccam-style optimizations.
    Directly processes the AXTree without building a separate TreeNode structure.
    """
    import copy
    import re

    # To avoid modifying the original tree, work on a copy
    tree_copy = copy.deepcopy(AX_tree)
    nodes = tree_copy["nodes"]

    # ============ Helper Functions ============
    def remove_unwanted_characters(text):
        """Clean text by removing special characters."""
        if not isinstance(text, str):
            return text
        text = text.replace('\xa0', ' ')
        cleaned_text = re.sub(r'[^\w\s,.!?;:\-\'"()&/\u2019@]+', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

    def get_node_name(node):
        """Safely get node name."""
        return node.get('name', {}).get('value', '').strip()

    def get_node_role(node):
        """Safely get node role."""
        return node.get('role', {}).get('value', '')

    def get_visible_children(node):
        """Get visible children of a node."""
        children = [node_id_to_node.get(cid) for cid in node.get('childIds', [])]
        return [c for c in children if c and not c.get('_is_hidden')]

    def all_children_invisible(node):
        """Check if all children are invisible."""
        children = [node_id_to_node.get(cid) for cid in node.get('childIds', [])]
        for c in children:
            if c and not c.get('_is_hidden'):
                return False
        return True

    def get_siblings(node):
        """Get siblings of a node (excluding itself)."""
        parent_id = node_id_to_parent_id.get(node['nodeId'])
        if not parent_id or parent_id not in node_id_to_node:
            return []
        parent_node = node_id_to_node[parent_id]
        siblings = []
        for cid in parent_node.get('childIds', []):
            if cid != node['nodeId'] and cid in node_id_to_node:
                siblings.append(node_id_to_node[cid])
        return siblings

    def extract_browsergym_id(node):
        """Extract browsergym_id from roledescription property."""
        for prop in node.get("properties", []):
            if prop.get("name") == "roledescription":
                value = prop.get("value", {}).get("value", "")
                if isinstance(value, str) and value.startswith("browsergym_id_"):
                    m = re.search(r'browsergym_id_(\w+)', value)
                    if m:
                        return m.group(1)
        # Fallback to the stored browsergym_id if already extracted
        return node.get("browsergym_id")

    # ============ Build Relationship Maps ============
    node_id_to_node = {node['nodeId']: node for node in nodes}
    node_id_to_parent_id = {}
    
    for node in nodes:
        node['_is_hidden'] = False
        # Extract and store browsergym_id for later use
        node['_browsergym_id'] = extract_browsergym_id(node)
        for child_id in node.get('childIds', []):
            if child_id in node_id_to_node:
                node_id_to_parent_id[child_id] = node['nodeId']

    # ============ Pass 1: Clean Text ============
    for node in nodes:
        if "name" in node and "value" in node["name"]:
            node["name"]["value"] = remove_unwanted_characters(node["name"]["value"])

    # ============ Pass 2: Remove Unwanted Properties ============
    UNWANTED_PROPS = {"focused", "autocomplete", "hasPopup", "expanded", 
                      "multiselectable", "orientation", "controls", 
                      "roledescription", "description"}
    
    for node in nodes:
        if "properties" in node:
            filtered = [p for p in node["properties"] if p.get("name") not in UNWANTED_PROPS]
            if filtered:
                node["properties"] = filtered
            else:
                del node["properties"]

    # ============ Pass 3: Remove Images ============
    for node in nodes:
        if node.get('_is_hidden'):
            continue
        node_role = get_node_role(node).lower()
        node_name = get_node_name(node)
        if all_children_invisible(node) and (node_role in ("img", "image") or node_name == "Image"):
            node['_is_hidden'] = True

    # ============ Pass 4: Remove Redundant Static Text ============
    def check_statictext_redundancy():
        for node in nodes:
            if node.get('_is_hidden'):
                continue
            
            node_role = get_node_role(node)
            if node_role not in ["StaticText", "LabelText", "caption"]:
                continue
            if not all_children_invisible(node):
                continue
            
            node_name = get_node_name(node)
            
            # Empty name
            if not node_name:
                node['_is_hidden'] = True
                continue
            
            # Name is contained in parent's name
            parent_id = node_id_to_parent_id.get(node['nodeId'])
            if parent_id and parent_id in node_id_to_node:
                parent_name = get_node_name(node_id_to_node[parent_id])
                if node_name and node_name in parent_name:
                    node['_is_hidden'] = True
                    continue
            
            # Name is contained in a sibling's name
            for sibling in get_siblings(node):
                sibling_name = get_node_name(sibling)
                if node_name and node_name in sibling_name:
                    node['_is_hidden'] = True
                    break

    check_statictext_redundancy()

    # ============ Pass 5: Merge Static Text to Parent ============
    for node in nodes:
        if node.get('_is_hidden'):
            continue
        
        node_role = get_node_role(node)
        if node_role not in ["StaticText", "LabelText", "caption"]:
            continue
        if not all_children_invisible(node):
            continue
        
        parent_id = node_id_to_parent_id.get(node['nodeId'])
        if not parent_id or parent_id not in node_id_to_node:
            continue
        
        parent_node = node_id_to_node[parent_id]
        parent_name = get_node_name(parent_node)
        
        # If parent has no name and this is the only child
        if not parent_name and len(parent_node.get('childIds', [])) == 1:
            if 'name' not in parent_node:
                parent_node['name'] = {}
            parent_node['name']['value'] = get_node_name(node)
            node['_is_hidden'] = True

    # Run redundancy check again after merging
    check_statictext_redundancy()

    # ============ Pass 6: Remove Fuzzy/Duplicate Nodes ============
    def is_node_differentiable(node):
        """Check if a node can be differentiated from its siblings."""
        if not all_children_invisible(node):
            return True
        
        parent_id = node_id_to_parent_id.get(node['nodeId'])
        if parent_id and parent_id in node_id_to_node:
            parent = node_id_to_node[parent_id]
            if get_node_role(parent) == "row":
                return True
        
        node_role = get_node_role(node)
        node_name = get_node_name(node)
        
        for sibling in get_siblings(node):
            if all_children_invisible(sibling):
                if get_node_role(sibling) == node_role and get_node_name(sibling) == node_name:
                    return False
        return True

    for node in nodes:
        if node.get('_is_hidden'):
            continue
        if all_children_invisible(node) and not is_node_differentiable(node):
            node['_is_hidden'] = True

    # ============ Pass 7: Replace Roles ============
    for node in nodes:
        if node.get('_is_hidden'):
            continue
        node_role = get_node_role(node)
        if node_role in ROLE_REPLACEMENT_DICT:
            node["role"]["value"] = ROLE_REPLACEMENT_DICT[node_role]

    # ============ Pass 8: Merge Menuitems and Options ============
    for node in nodes:
        if node.get('_is_hidden'):
            continue
        
        visible_children = get_visible_children(node)
        if not visible_children:
            continue
        
        child_roles = [get_node_role(c) for c in visible_children]
        
        if all(r == 'menuitem' for r in child_roles) or all(r == 'option' for r in child_roles):
            child_names = [get_node_name(c) for c in visible_children]
            current_name = get_node_name(node)
            joined_names = "; ".join(filter(None, child_names))
            
            final_name = f"{current_name}: {joined_names}" if current_name else joined_names
            
            if 'name' not in node:
                node['name'] = {}
            node['name']['value'] = final_name
            
            for c in visible_children:
                c['_is_hidden'] = True

    # ============ Pass 9: Merge Description Lists ============
    for node in nodes:
        if node.get('_is_hidden') or get_node_role(node) != 'DescriptionList':
            continue
        
        visible_children = get_visible_children(node)
        
        # First: merge grandchildren up into Details
        for child in visible_children:
            if get_node_role(child) == 'DescriptionListDetail' and not get_node_name(child):
                visible_grandchildren = get_visible_children(child)
                if len(visible_grandchildren) == 1:
                    gc = visible_grandchildren[0]
                    if 'name' not in child:
                        child['name'] = {}
                    child['name']['value'] = get_node_name(gc)
                    gc['_is_hidden'] = True
        
        # Second: merge Term/Detail pairs
        def reformat_sublist(buffer):
            if len(buffer) > 1:
                term = buffer[0]
                details = buffer[1:]
                detail_names = [get_node_name(d) for d in details]
                term_name = get_node_name(term)
                joined = "; ".join(filter(None, detail_names))
                
                if 'name' not in term:
                    term['name'] = {}
                term['name']['value'] = f"{term_name}: {joined}"
                
                for d in details:
                    d['_is_hidden'] = True
        
        visible_children = get_visible_children(node)  # Re-fetch after modifications
        buffer = []
        for child in visible_children:
            child_role = get_node_role(child)
            has_visible_children = bool(get_visible_children(child))
            
            if child_role == 'DescriptionListTerm' and not has_visible_children:
                reformat_sublist(buffer)
                buffer = [child]
            elif child_role == 'DescriptionListDetail' and not has_visible_children and buffer:
                buffer.append(child)
            else:
                reformat_sublist(buffer)
                buffer = []
        reformat_sublist(buffer)

    # ============ Pass 10: Merge Duplicated Headings ============
    for node in nodes:
        if node.get('_is_hidden') or not all_children_invisible(node):
            continue
        
        parent_id = node_id_to_parent_id.get(node['nodeId'])
        if not parent_id or parent_id not in node_id_to_node:
            continue
        
        parent = node_id_to_node[parent_id]
        visible_siblings = [s for s in get_siblings(node) if not s.get('_is_hidden')]
        if visible_siblings:
            continue
        
        node_role = get_node_role(node)
        parent_role = get_node_role(parent)
        node_name = get_node_name(node)
        parent_name = get_node_name(parent)
        
        if node_name == parent_name:
            if node_role == "heading" and parent_role not in UNINTERACTIVE_ROLES:
                node['_is_hidden'] = True
            elif parent_role == "heading" and node_role not in UNINTERACTIVE_ROLES:
                parent["role"]["value"] = node_role
                parent['_browsergym_id'] = node.get('_browsergym_id')
                node['_is_hidden'] = True

    # ============ Pass 11: Table Reformatting ============
    def format_table_row(row_node):
        """Format a table row as markdown."""
        visible_cells = get_visible_children(row_node)
        cell_contents = []
        for cell in visible_cells:
            # Collect all text content from the cell
            cell_text = get_node_name(cell)
            if not cell_text:
                # Try to get text from children
                texts = []
                def collect_text(n):
                    if n.get('_is_hidden'):
                        return
                    name = get_node_name(n)
                    if name:
                        texts.append(name)
                    for cid in n.get('childIds', []):
                        if cid in node_id_to_node:
                            collect_text(node_id_to_node[cid])
                collect_text(cell)
                cell_text = " ".join(texts)
            cell_contents.append(cell_text)
        return cell_contents

    for node in nodes:
        if node.get('_is_hidden') or get_node_role(node) != 'table':
            continue
        
        visible_children = get_visible_children(node)
        rows = [c for c in visible_children if get_node_role(c) == 'row']
        rowgroups = [c for c in visible_children if get_node_role(c) == 'rowgroup']
        
        # Collect all rows (from direct children and rowgroups)
        all_rows = rows[:]
        for rg in rowgroups:
            rg_rows = [c for c in get_visible_children(rg) if get_node_role(c) == 'row']
            all_rows.extend(rg_rows)
        
        if not all_rows:
            continue
        
        # Format as markdown table
        table_lines = []
        for i, row in enumerate(all_rows):
            cells = format_table_row(row)
            if cells:
                row_str = "| " + " | ".join(cells) + " |"
                if 'name' not in row:
                    row['name'] = {}
                row['name']['value'] = row_str
                
                # Hide cell children
                for cell in get_visible_children(row):
                    cell['_is_hidden'] = True
                
                # Add separator after header row
                if i == 0 and len(all_rows) > 1:
                    first_row_cells = get_visible_children(all_rows[0])
                    if any(get_node_role(c) == 'columnheader' for c in first_row_cells):
                        sep = "| " + " | ".join(["---"] * len(cells)) + " |"
                        # Store separator info (will be handled in output)
                        row['_table_separator'] = sep

    # ============ Final Output Generation ============
    def dfs_output(node_id, depth):
        node = node_id_to_node.get(node_id)
        if not node:
            return ""
        
        # If hidden, process children at same depth
        if node.get('_is_hidden'):
            child_strs = [dfs_output(cid, depth) for cid in node.get('childIds', [])]
            return "\n".join(filter(None, child_strs))
        
        node_role = get_node_role(node)
        node_name = get_node_name(node)
        
        # Skip empty generic/none nodes
        if (node_role in ("none", "generic", "") or not node_role) and not node_name:
            child_strs = [dfs_output(cid, depth) for cid in node.get('childIds', [])]
            return "\n".join(filter(None, child_strs))
        
        indent = "\t" * depth
        bid = node.get('_browsergym_id')
        
        # Build the node string
        parts = []
        
        # Only show bid for interactive roles
        if bid and node_role not in UNINTERACTIVE_ROLES and node_role not in set(ROLE_REPLACEMENT_DICT.values()):
            parts.append(f'[{bid}]')
        
        parts.append(node_role)
        
        if node_name:
            parts.append(repr(node_name))
        
        # Add value if present
        if 'value' in node and 'value' in node.get('value', {}):
            parts.append(f'value={repr(node["value"]["value"])}')
        
        # Add properties
        prop_strs = []
        for prop in node.get("properties", []):
            prop_name = prop.get("name")
            if 'value' in prop and 'value' in prop['value']:
                prop_value = prop['value']['value']
                if isinstance(prop_value, bool):
                    if prop_value:
                        prop_strs.append(prop_name)
                else:
                    prop_strs.append(f'{prop_name}: {prop_value}')
        
        if prop_strs:
            parts.append("[" + ", ".join(prop_strs) + "]")
        
        node_str = indent + " ".join(parts)
        
        # Handle table separator
        result_lines = [node_str]
        if node.get('_table_separator'):
            result_lines.append(indent + node['_table_separator'])
        
        # Recurse for children
        child_strs = [dfs_output(cid, depth + 1) for cid in node.get('childIds', [])]
        result_lines.extend([s for s in child_strs if s])
        
        return "\n".join(result_lines)

    # Find root nodes and generate output
    if nodes:
        all_child_ids = set(cid for n in nodes for cid in n.get('childIds', []))
        root_nodes = [n for n in nodes if n['nodeId'] not in all_child_ids]
        
        if root_nodes:
            return "\n".join(filter(None, [dfs_output(root['nodeId'], 0) for root in root_nodes]))
        elif nodes:
            return dfs_output(nodes[0]['nodeId'], 0)
    
    return ""
