# 🤖 Browser Agent

An intelligent browser automation system for complex dynamic web environments, featuring site-specific knowledge adaptation and progressive progress summarization for robust, long-horizon task execution.

## 📋 Overview

Browser Agent is an intelligent web automation framework developed for the [BrowserGym](https://github.com/ServiceNow/BrowserGym) environment that systematically addresses three critical bottlenecks in real-world web automation:
- **Site heterogeneity** - through declarative knowledge bases
- **Long-horizon stability** - through progressive memory compression
- **Observation quality** - through multimodal perception fusion

### 🏛️ Core Architecture

The system implements a **Perception–Reasoning–Actuation** closed-loop with dual-agent collaboration:

- **Operator Agent**: Executes web actions based on site-specific knowledge and multimodal observations
- **Summarizer Agent**: Compresses trajectory history into fixed-length progress summaries with conditional corrective guidance

This architecture enables context-efficient decision making: reducing prompt complexity from O(T) to O(1) while maintaining task coherence over 10-20+ interaction steps.

### 🗂️ Site-Specific Knowledge Adaptation

Real-world web environments contain numerous site-specific quirks and platform defects. Our system addresses this through:

- **Tips Repository**: Maintains domain-specific navigation patterns, interaction guidelines, and common pitfalls for each target site (GitLab, Reddit, Shopping, Map, etc.)
- **Action Mapping Layer**: Automatically detects site characteristics and transparently maps model-generated "standard actions" to site-executable equivalents
- **Declarative Adaptation Rules**: URL/title pattern matching + rule-based substitution for transparent handling of environment defects

**Example - GitLab Global Search Mapping**:
- **Problem**: Global search submission fails in GitLab test environment
- **Solution**: Automatically maps `fill(search_box, keyword)` → `goto('/search?search=keyword&nav_source=navbar')`
- **Benefit**: Model operates "as if on a normal site" without awareness of underlying adaptation

### 📝 Progressive Progress Summarization

For long-horizon tasks facing context window limitations and decision drift:

- **Updateable Compressed Memory**: Linearly growing full trajectories → fixed-length progress summaries
- **Conditional Correction Mechanism**: Detects deviation/errors and provides 1-2 actionable suggestions only when necessary
- **Structured Summary Template**:
  - ✅ **Current Progress**: Completed sub-goals and extracted key facts
  - 🔍 **Current State Analysis**: Page status and interactable elements
  - ➡️ **Next-step Guidance** (conditional): Corrective suggestions only when deviation detected

### 👁️ Multimodal Observation

Hybrid perception strategy combining:

- **Accessibility Tree (AXTree)**: Structured semantic representation of interactable elements (type, state, actionable anchors)
- **Set-of-Mark (SoM)**: Bounding box annotations overlaid on screenshots for vision-structure alignment
- **Flexible Configuration**: Supports combinations of AXTree/DOM/Screenshot/SoM for different scenarios

### 📊 Performance

Our approach achieves **71.18% task success rate** on the [WebArena](https://arxiv.org/abs/2307.13854) benchmark with the **GPT-5 model**, demonstrating superior long-horizon stability and site adaptation capabilities.

For detailed trajectory data and analysis, see [Trajectory Results Documentation](trajectories/README.md).


## ✨ Key Features

### 🏗️ Systematic Engineering
- **Site Knowledge Decoupling**: Externalizes site-specific logic to avoid polluting general prompts
- **Extensibility**: New sites require only adding tips files and adaptation rules
- **Maintainability**: Centralized site-specific knowledge management with version control
- **Observability**: Records adaptation rule hits to quantify knowledge base effectiveness

### 🧠 Long-Horizon Stability
- **Memory Compression**: Breakthrough context window limitations for 10-20+ step complex tasks
- **Decision Coherence**: Prevents decision drift through progressive summarization
- **Self-Correction**: Conditional correction mechanism detects and fixes goal-deviating behaviors

### 🔄 Intelligent Adaptation
- **Transparent Mapping**: Models operate on "standardized" actions while system handles site quirks
- **Evidence-Based Summarization**: Summaries must be grounded in actual observations (screenshots/AXTree)
- **Dual-Module Collaboration**: Operator (execution) + Summarizer (progress tracking) form closed loop

---
> **Note**: This agent system is developed as a prototype based on the [WebArena](https://arxiv.org/abs/2307.13854) environment, demonstrating production-viable engineering implementations that can be adapted for real-world deployment scenarios.

---

> ⚠️ **Source Code Status**: The implementation code in the `agent/` directory is currently being organized. The complete source code will be made available soon. Stay tuned for updates!


## 🚀 Quick Start

### 📦 Requirements
- Python == 3.12
- UV package manager

### 1️⃣ Create Virtual Environment (Optional)

```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 2️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

This will automatically install:
- Python third-party libraries (numpy, Pillow, scikit-image, etc.)
- BrowserGym local packages (core, experiments, webarena)

After installing dependencies, install Playwright's Chromium browser:

```bash
playwright install chromium
```

### 3️⃣ Configure Environment Variables

Copy the example configuration file and modify it with your settings:

```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

Example `.env` configuration:

```bash
# API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=your_api_base_url

# WebArena Configuration
WEBARENA_SHOPPING_URL=http://localhost:7770
WEBARENA_REDDIT_URL=http://localhost:9999
# ... other configurations
```

For WebArena Docker environment setup, please refer to the [official WebArena documentation](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md).


### 4️⃣ Usage
**Running WebArena Tasks**
```
python agent/run_webarena.py \
    --task shopping_admin \
    --task_ids 3 \
    --exp demo \
    --model_name gpt-5
```

**Adding Experiential Knowledge**

1. Create site-specific knowledge files in `agent/tips/`
2. Format: Plain text with navigation patterns, interaction guidelines, and common pitfalls
3. Enable with `--tips true` flag

Example tips structure:

```text
# GitLab Tips
- Use direct URL navigation for global search: /search?search=keyword
- Issue creation requires explicit confirmation button click
- Project permissions are hierarchical: Owner > Maintainer > Developer
```


**Custom Observation Strategies**
Combine different observation modalities by adjusting flags:
- Pure visual: --use_screenshot True --use_axtree False
- Hybrid: --use_screenshot True --use_axtree True --use_som True
- Structure-only: --use_html True --use_screenshot False


## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This project builds upon:
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - Web automation framework
- [WebArena](https://github.com/web-arena-x/webarena.git) - Benchmark for web agents

Special thanks to the open-source community for their foundational work in web automation and reinforcement learning.

## 📧 Contact

For questions, collaboration, or research inquiries, please:
- Open an issue on GitHub
- Submit a pull request
- Contact the authors

---

**Note**: This is a research prototype demonstrating production-ready engineering practices. Contributions and feedback are welcome!
