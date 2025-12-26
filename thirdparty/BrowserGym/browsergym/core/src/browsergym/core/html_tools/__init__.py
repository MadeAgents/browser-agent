# =============================================================================
# NEW MODULE ADDED:
# This module is newly added to the BrowserGym project
# Creation Date: 2025-12-22
# Purpose:
#   - Provides HTML parsing and processing tools
#   - Exports HtmlParser, IdentifierTool, and HtmlPrompt classes
#   - Supports element identification and labeling
#   - Enables structured HTML observation generation
# =============================================================================

from .identifier import IdentifierTool
from .prompt import HtmlPrompt
from .html_parser import HtmlParser

from .utils import print_html_object
from .configs import basic_attrs, mind2web_keep_attrs