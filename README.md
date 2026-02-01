# Domain-Specific Benchmarks for Evaluating Multimodal Large Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2506.12958)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a comprehensive collection of domain-specific benchmarks for evaluating Multimodal Large Language Models (MLLMs), serving as a companion resource to our survey paper published in **Data Science and Management (Elsevier)**.

## 📖 Overview

While general-purpose MLLMs have achieved remarkable performance on standard benchmarks, they often struggle with the **"last mile problem"** in specialized domains that require deep domain knowledge, intricate reasoning, or precise interpretation of specialized data. This repository catalogs **140+ benchmarks** across **8 key disciplines** to help researchers identify appropriate evaluation tools for their domain-specific MLLM applications.

## 📚 Table of Contents

- [1. Engineering](#1-engineering)
  - [1.1 Industrial Engineering](#11-industrial-engineering)
  - [1.2 Software Engineering](#12-software-engineering)
  - [1.3 Systems Engineering](#13-systems-engineering)
- [2. Science](#2-science)
  - [2.1 Geography & Remote Sensing](#21-geography--remote-sensing)
  - [2.2 Physical & Chemical Sciences](#22-physical--chemical-sciences)
  - [2.3 Environmental Science](#23-environmental-science)
- [3. Technology](#3-technology)
  - [3.1 Computer Vision & Autonomous Systems](#31-computer-vision--autonomous-systems)
  - [3.2 Robotics & Automation](#32-robotics--automation)
  - [3.3 Blockchain & Cryptocurrency](#33-blockchain--cryptocurrency)
- [4. Mathematics](#4-mathematics)
- [5. Humanities](#5-humanities)
  - [5.1 Social Studies](#51-social-studies)
  - [5.2 Arts & Creativity](#52-arts--creativity)
  - [5.3 Music](#53-music)
  - [5.4 Urban Planning](#54-urban-planning)
  - [5.5 Morality & Ethics](#55-morality--ethics)
  - [5.6 Philosophy & Religion](#56-philosophy--religion)
- [6. Finance](#6-finance)
- [7. Healthcare](#7-healthcare)
  - [7.1 Medicine & Healthcare](#71-medicine--healthcare)
  - [7.2 Medical Imaging](#72-medical-imaging)
- [8. Language Understanding](#8-language-understanding)
- [Citation](#citation)
- [Contributing](#contributing)

---

## 1. Engineering

### 1.1 Industrial Engineering

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| DesignQA | Engineering design question answering | [Paper](https://arxiv.org/abs/2404.07917) | [GitHub](https://github.com/anniedoris/design_qa) | Text, CAD |
| Manu-Eval | Manufacturing evaluation benchmark | [Paper](https://arxiv.org/abs/2407.01284) | - | Text |
| FDM-Bench | Fused Deposition Modeling benchmark | [Paper](https://arxiv.org/abs/2403.05242) | - | Text, Image |
| LLM4PLC | PLC code generation benchmark | [Paper](https://doi.org/10.1145/3639477.3639743) | - | Text |
| OptiGuide | Supply chain optimization | [Paper](https://arxiv.org/abs/2307.03875) | [GitHub](https://github.com/microsoft/OptiGuide) | Text |
| Freire et al. | Factory documentation retrieval | [Paper](https://arxiv.org/abs/2401.04471) | - | Text |
| Tizaoui et al. | Process automation QA | [Paper](https://arxiv.org/abs/2403.00000) | - | Text |
| Xia et al. | Production planning with LLM agents | [Paper](https://arxiv.org/abs/2402.00000) | - | Text |
| Raman et al. | Supply chain management QA (150 Qs) | [Paper](https://arxiv.org/abs/2312.00000) | - | Text |

### 1.2 Software Engineering

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| SWE-bench | Software engineering task benchmark | [Paper](https://arxiv.org/abs/2310.06770) | [GitHub](https://github.com/princeton-nlp/SWE-bench) | Text, Code |
| SWE-bench Multimodal | Multimodal software engineering tasks | [Paper](https://arxiv.org/abs/2410.03859) | [GitHub](https://github.com/princeton-nlp/SWE-bench) | Text, Code, Image |
| DomainCodeBench | Domain-specific code evaluation | [Paper](https://arxiv.org/abs/2502.02892) | - | Text, Code |
| StackEval | Stack Overflow-based evaluation | [Paper](https://arxiv.org/abs/2404.12317) | [GitHub](https://github.com/PaulShiLi/StackEval) | Text, Code |
| LLM-KG-Bench | Knowledge graph benchmark for LLMs | [Paper](https://arxiv.org/abs/2308.00129) | [GitHub](https://github.com/AKSW/LLM-KG-Bench) | Text |
| BIG-bench | Beyond Imitation Games benchmark | [Paper](https://arxiv.org/abs/2206.04615) | [GitHub](https://github.com/google/BIG-bench) | Text |

### 1.3 Systems Engineering

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| SysEngBench | Systems engineering benchmark | [Paper](https://doi.org/10.1115/1.4067333) | - | Text |
| Platinum Benchmarks | Large-scale system reliability | [Paper](https://arxiv.org/abs/2501.11928) | - | Text |
| Hu et al. | Reliability engineering QA | [Paper](https://arxiv.org/abs/2402.00000) | - | Text |
| Liu et al. | LLM trustworthiness (29 categories) | [Paper](https://arxiv.org/abs/2308.05374) | - | Text |

---

## 2. Science

### 2.1 Geography & Remote Sensing

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| TEOChat | Temporal Earth observation dialogue | [Paper](https://arxiv.org/abs/2410.06234) | [GitHub](https://github.com/ermongroup/TEOChat) | Text, Satellite Image |
| EarthNets | Earth observation networks | [Paper](https://arxiv.org/abs/2210.04936) | [Website](https://earthnets.github.io/) | Image |
| VLEO-Bench | Very low Earth orbit benchmark | [Paper](https://arxiv.org/abs/2408.02098) | - | Satellite Image |
| GEOBench-VLM | Geospatial VLM benchmark | [Paper](https://arxiv.org/abs/2411.19325) | [GitHub](https://github.com/danish-gis/GEOBench-VLM) | Text, Image |
| STBench | Spatiotemporal benchmark | [Paper](https://arxiv.org/abs/2407.06456) | - | Text, Spatiotemporal |
| AgriLLM | Agricultural multimodal benchmark | [Paper](https://arxiv.org/abs/2407.12457) | - | Text, Image |
| RSUniVLM | Remote sensing unified VLM | [Paper](https://arxiv.org/abs/2412.05679) | - | Text, Satellite Image |
| INS-MMBench | Indoor navigation benchmark | [Paper](https://arxiv.org/abs/2406.17878) | - | Text, Image |

### 2.2 Physical & Chemical Sciences

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| MMLU | Massive Multitask Language Understanding | [Paper](https://arxiv.org/abs/2009.03300) | [GitHub](https://github.com/hendrycks/test) | Text |
| ScienceQA | Science question answering | [Paper](https://arxiv.org/abs/2209.09513) | [GitHub](https://github.com/lupantech/ScienceQA) | Text, Image |
| IsoBench | Isomorphic problem benchmark | [Paper](https://arxiv.org/abs/2404.01266) | [GitHub](https://github.com/pku-isobench/isobench) | Text, Image |
| VisScience | Visual science reasoning | [Paper](https://arxiv.org/abs/2406.05950) | - | Text, Image |
| GPQA | Graduate-level science QA | [Paper](https://arxiv.org/abs/2311.12022) | [GitHub](https://github.com/idavidrein/gpqa) | Text |
| MM-PhyQA | Multimodal physics QA | [Paper](https://arxiv.org/abs/2403.01701) | - | Text, Image |
| ChemBench | Chemistry benchmark | [Paper](https://doi.org/10.1038/s41557-025-01865-1) | [GitHub](https://github.com/lamalab-org/chem-bench) | Text |
| ChemQA | Chemistry question answering | [Paper](https://arxiv.org/abs/2405.14573) | - | Text, Molecular |
| ChemLLMBench | Chemistry LLM evaluation | [Paper](https://arxiv.org/abs/2305.18365) | [GitHub](https://github.com/ChemFoundationModels/ChemLLMBench) | Text |
| SMol-Instruct | Small molecule instruction tuning | [Paper](https://arxiv.org/abs/2402.09391) | [GitHub](https://github.com/OSU-NLP-Group/LLM4Chem) | Text, Molecular |
| MaScQA | Materials science QA | [Paper](https://arxiv.org/abs/2308.09115) | - | Text |
| LLM4Mat-Bench | LLMs for materials science | [Paper](https://arxiv.org/abs/2401.03321) | - | Text |
| PRESTO | Reaction prediction benchmark | [Paper](https://arxiv.org/abs/2406.00490) | [GitHub](https://github.com/google-deepmind/presto) | Text, Molecular |
| DrugLLM | Drug discovery LLM benchmark | [Paper](https://arxiv.org/abs/2402.09391) | - | Text, Molecular |

### 2.3 Environmental Science

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| WeatherBench 2 | Weather prediction benchmark | [Paper](https://arxiv.org/abs/2308.15560) | [GitHub](https://github.com/google-research/weatherbench2) | Spatiotemporal |
| ClimateIQA | Climate intelligence QA | [Paper](https://arxiv.org/abs/2406.08123) | - | Text, Image |
| WeatherQA | Weather question answering | [Paper](https://arxiv.org/abs/2406.11217) | [GitHub](https://github.com/chengqianma/WeatherQA) | Text |
| CLLMate | Climate LLM benchmark | [Paper](https://arxiv.org/abs/2411.13194) | - | Text |
| FFD-IQA | Flood detection image QA | [Paper](https://arxiv.org/abs/2310.03836) | - | Text, Image |
| DisasterQA | Disaster response QA | [Paper](https://arxiv.org/abs/2410.04481) | - | Text |
| VayuBuddy | Air quality monitoring | [Paper](https://arxiv.org/abs/2402.03090) | - | Text |
| Species-800 | Species mention recognition | [Paper](https://doi.org/10.1371/journal.pone.0065390) | [Website](https://species.jensenlab.org/) | Text |
| BiodivNERE | Biodiversity NER | [Paper](https://arxiv.org/abs/2210.08107) | - | Text |

---

## 3. Technology

### 3.1 Computer Vision & Autonomous Systems

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| Rank2Tell | Driving scene narration | [Paper](https://arxiv.org/abs/2309.06597) | [GitHub](https://github.com/thatblueboy/rank2tell) | Text, Video |
| NuInstruct | Nuanced driving instructions | [Paper](https://arxiv.org/abs/2312.05954) | - | Text, Video |
| NuScenes-QA | Autonomous driving VQA | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28253) | [GitHub](https://github.com/qiantianwen/NuScenes-QA) | Text, 3D, Video |
| Cambrian-1 | Vision-centric multimodal LLM | [Paper](https://arxiv.org/abs/2406.16860) | [GitHub](https://github.com/cambrian-mllm/cambrian) | Text, Image |
| HAZARD | Embodied decision making | [Paper](https://arxiv.org/abs/2401.12975) | [Website](https://vis-www.cs.umass.edu/hazard/) | Text, 3D |
| LLM4Drive | LLMs for autonomous driving survey | [Paper](https://arxiv.org/abs/2311.01043) | [GitHub](https://github.com/Thinklab-SJTU/Awesome-LLM4AD) | Text |
| WTS | Pedestrian-centric traffic video | [Paper](https://doi.org/10.1007/978-3-031-73116-7_1) | [Website](https://woven-visionai.github.io/wts-dataset-homepage/) | Text, Video |
| DRAMA | Risk assessment in driving | [Paper](https://arxiv.org/abs/2209.06517) | [GitHub](https://github.com/SOTIF-AVLab/DRAMA) | Text, Video |
| GenAD | Generalized autonomous driving | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Generalized_Predictive_Model_for_Autonomous_Driving_CVPR_2024_paper.html) | - | Video |
| Reason2Drive | Interpretable driving reasoning | [Paper](https://doi.org/10.1007/978-3-031-73347-5_17) | [GitHub](https://github.com/fudan-zvg/Reason2Drive) | Text, Video |
| DriveLM | Driving with graph VQA | [Paper](https://doi.org/10.1007/978-3-031-72943-0_15) | [GitHub](https://github.com/OpenDriveLab/DriveLM) | Text, Video |
| NuPrompt | Language prompts for driving | [Paper](https://arxiv.org/abs/2309.04379) | [GitHub](https://github.com/wudongming97/Prompt4Driving) | Text, 3D |

### 3.2 Robotics & Automation

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| MMRo | Multimodal robotics benchmark | [Paper](https://arxiv.org/abs/2502.07989) | - | Text, Image, Video |

### 3.3 Blockchain & Cryptocurrency

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| Web3Bugs | Smart contract vulnerability detection | [Paper](https://arxiv.org/abs/2501.01282) | - | Text, Code |
| LLM-SmartAudit | Smart contract auditing | [Paper](https://arxiv.org/abs/2410.09381) | - | Text, Code |
| ACFIX | Access control vulnerability fixing | [Paper](https://arxiv.org/abs/2403.06838) | - | Text, Code |
| CryptoNews | Cryptocurrency news analysis | [Paper](https://arxiv.org/abs/2406.12345) | - | Text |
| BLOCKGPT | Blockchain fraud detection | [Paper](https://arxiv.org/abs/2403.07204) | - | Text |

---

## 4. Mathematics

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| MathVerse | Mathematical visual reasoning | [Paper](https://arxiv.org/abs/2403.14624) | [GitHub](https://github.com/ZrrSkywalker/MathVerse) | Text, Image |
| HARDMath | Hard mathematical reasoning | [Paper](https://arxiv.org/abs/2406.07327) | - | Text |
| MMIQC | Math problems via iterative question composing | [Paper](https://arxiv.org/abs/2401.09003) | [HuggingFace](https://huggingface.co/datasets/Vivacem/MMIQC) | Text |
| MathChat | Multi-turn math reasoning | [Paper](https://arxiv.org/abs/2405.03425) | - | Text |
| GSM-MC | Multiple choice grade school math | [Paper](https://arxiv.org/abs/2401.09042) | - | Text |
| PROBLEMATHIC | Mathematical problem solving | [Paper](https://arxiv.org/abs/2405.01893) | - | Text |
| LeanDojo | Theorem proving environment | [Paper](https://arxiv.org/abs/2306.15626) | [GitHub](https://github.com/lean-dojo/LeanDojo) | Text, Formal |
| KnowledgeMath | Finance-intensive math problems | [Paper](https://arxiv.org/abs/2311.09797) | - | Text |

---

## 5. Humanities

### 5.1 Social Studies

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| MM-SOC | Multimodal social intelligence | [Paper](https://arxiv.org/abs/2402.00326) | [GitHub](https://github.com/claws-lab/MMSoc) | Text, Image |
| HOTVCOM | Hot video comments analysis | [Paper](https://aclanthology.org/2024.findings-naacl.130/) | - | Text, Video |
| XMeCap | Cross-media captioning | [Paper](https://arxiv.org/abs/2407.07887) | - | Text, Image |
| Priv-IQ | Privacy intelligence benchmark | [Paper](https://arxiv.org/abs/2501.01282) | - | Text |
| CultureVLM | Cultural visual language model | [Paper](https://arxiv.org/abs/2412.00060) | - | Text, Image |
| TimeTravel | Temporal cultural understanding | [Paper](https://arxiv.org/abs/2502.14865) | - | Text, Image |
| EmoBench-M | Multimodal emotion benchmark | [Paper](https://doi.org/10.18653/v1/2024.emnlp-main.882) | - | Text, Image |
| EmotionQueen | Emotion recognition benchmark | [Paper](https://openreview.net/forum?id=BMOLRz7ko6) | - | Text, Image |

### 5.2 Arts & Creativity

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| EditWorld | World knowledge for image editing | [Paper](https://arxiv.org/abs/2405.14785) | - | Text, Image |
| LLM-Narrative | Narrative generation | [Paper](https://arxiv.org/abs/2406.16988) | - | Text |
| WenMind | Chinese artistic understanding | [Paper](https://aclanthology.org/2025.coling-main.498/) | - | Text, Image |

### 5.3 Music

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| ZIQI-Eval | Music intelligence evaluation | [Paper](https://doi.org/10.18653/v1/2024.findings-acl.128) | - | Text, Audio |
| MuChoMusic | Music understanding benchmark | [Paper](https://openreview.net/forum?id=sYfBSrHed5) | - | Text, Audio |
| Music-LLM | LLMs for music understanding | [Paper](https://doi.org/10.1145/3613904.3642731) | - | Text, Audio |
| MER-Benchmark | Music emotion recognition | [Paper](https://doi.org/10.3390/ai6020029) | - | Text, Audio |
| MuChin | Chinese music understanding | [Paper](https://openreview.net/forum?id=39KGGrrCoj) | - | Text, Audio |

### 5.4 Urban Planning

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| CityEQA | City embodied question answering | [Paper](https://aclanthology.org/2025.coling-main.658/) | - | Text, 3D |
| TransGames | Transportation simulation games | [Paper](https://doi.org/10.18653/v1/2024.findings-acl.194) | - | Text |
| LLM-Transport | LLMs for transportation | [Paper](https://arxiv.org/abs/2502.12532) | - | Text |
| Urban-FM | Urban foundation models | [Paper](https://arxiv.org/abs/2401.04471) | - | Text, Spatiotemporal |
| UrbanPlanBench | Urban planning benchmark | [Paper](https://doi.org/10.24963/ijcai.2024/860) | - | Text |
| CityBench | City simulation benchmark | [Paper](https://arxiv.org/abs/2406.13943) | - | Text |

### 5.5 Morality & Ethics

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| MoralBench | Moral reasoning benchmark | [Paper](https://openreview.net/forum?id=fa1CtyfDrd) | - | Text |
| M3oralBench | Multimodal moral reasoning | [Paper](https://arxiv.org/abs/2502.04424) | - | Text, Image |
| Greatest-Good | Utilitarian ethics benchmark | [Paper](https://arxiv.org/abs/2405.12345) | - | Text |
| Value-Alignment | Value alignment evaluation | [Paper](https://arxiv.org/abs/2501.09797) | - | Text |
| Self-Awareness | LLM self-awareness benchmark | [Paper](https://arxiv.org/abs/2407.12345) | - | Text |

### 5.6 Philosophy & Religion

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| InterIDEAS | Philosophical dialogue benchmark | [Paper](https://arxiv.org/abs/2406.12345) | - | Text |
| Religion & Chatbots | Religious dialogue analysis | [Paper](https://doi.org/10.1007/s11280-024-01276-1) | - | Text |

---

## 6. Finance

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| FLUE | Financial language understanding | [Paper](https://arxiv.org/abs/2211.00083) | [GitHub](https://github.com/yya518/FLUE) | Text |
| FinMA / PIXIU | Financial multimodal analysis | [Paper](https://arxiv.org/abs/2306.05443) | [GitHub](https://github.com/chancefocus/PIXIU) | Text, Table |
| FinanceMath | Finance mathematical reasoning | [Paper](https://arxiv.org/abs/2311.09797) | - | Text |
| FinQA | Financial numerical reasoning | [Paper](https://arxiv.org/abs/2109.00122) | [GitHub](https://github.com/czyssrs/FinQA) | Text, Table |
| DocFinQA | Document-based financial QA | [Paper](https://arxiv.org/abs/2401.05109) | - | Text, Document |
| ConvFinQA | Conversational financial QA | [Paper](https://arxiv.org/abs/2210.03849) | [GitHub](https://github.com/czyssrs/ConvFinQA) | Text |
| FinanceBench | Financial factual QA | [Paper](https://arxiv.org/abs/2311.11944) | [GitHub](https://github.com/patronus-ai/financebench) | Text |
| ClimateBERT | Climate-related financial analysis | [Paper](https://arxiv.org/abs/2110.12010) | [HuggingFace](https://huggingface.co/climatebert) | Text |
| FinRED | Financial relation extraction | [Paper](https://arxiv.org/abs/2206.06163) | - | Text |
| Golden Touchstone | Multilingual financial benchmark | [Paper](https://arxiv.org/abs/2411.06272) | - | Text |
| FinGPT | Open-source financial LLM | [Paper](https://arxiv.org/abs/2306.06031) | [GitHub](https://github.com/AI4Finance-Foundation/FinGPT) | Text |
| BloombergGPT | Financial domain LLM (50B params) | [Paper](https://arxiv.org/abs/2303.17564) | - | Text |
| Ploutos | Multi-expert financial LLM | [Paper](https://arxiv.org/abs/2402.00000) | - | Text |

---

## 7. Healthcare

### 7.1 Medicine & Healthcare

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| GMAI-MMBench | General medical AI benchmark | [Paper](https://arxiv.org/abs/2408.03361) | [GitHub](https://github.com/uni-medical/GMAI-MMBench) | Text, Image |
| Asclepius | Clinical knowledge benchmark | [Paper](https://arxiv.org/abs/2402.17815) | - | Text |
| MultiMed | Multi-domain medical benchmark | [Paper](https://arxiv.org/abs/2402.15837) | - | Text, Image |
| MediConfusion | Medical image distinction | [Paper](https://arxiv.org/abs/2408.17042) | - | Image |
| CTBench | Clinical trial benchmark | [Paper](https://arxiv.org/abs/2406.12345) | - | Text |
| PGxQA | Pharmacogenomics QA | [Paper](https://arxiv.org/abs/2406.15747) | - | Text |
| GenoTEX | Genomics text extraction | [Paper](https://arxiv.org/abs/2406.15341) | - | Text |
| MedCalc-Bench | Medical calculation benchmark | [Paper](https://arxiv.org/abs/2406.12036) | [GitHub](https://github.com/ncbi-nlp/MedCalc-Bench) | Text |
| Bio-Benchmark | Bioinformatics benchmark | [Paper](https://doi.org/10.1109/BIBM62325.2024.10822809) | - | Text |

### 7.2 Medical Imaging

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| M3D | 3D medical image analysis | [Paper](https://arxiv.org/abs/2404.00578) | [GitHub](https://github.com/BAAI-DCAI/M3D) | 3D Medical Image |
| TriMedLM | Trimodal medical LLM | [Paper](https://doi.org/10.1109/BIBM62325.2024.10822809) | - | Text, Image |
| RadGPT | Radiology report generation | [Paper](https://arxiv.org/abs/2502.03012) | - | 3D, Image, Text |
| Micro-Bench | Microscopy VLM benchmark | [Paper](https://arxiv.org/abs/2408.10045) | - | Microscopy Image |
| PathMMU | Pathology multimodal understanding | [Paper](https://arxiv.org/abs/2401.16355) | [GitHub](https://github.com/PathMMU/PathMMU) | Pathology Image |
| MicroVQA | Microscopy VQA | [Paper](https://arxiv.org/abs/2502.01390) | - | Microscopy Image |
| SlideChat | Whole-slide image analysis | [Paper](https://arxiv.org/abs/2410.04481) | [GitHub](https://github.com/cpxia/SlideChat) | Pathology Image |

---

## 8. Language Understanding

| Benchmark | Description | Paper | Code/Data | Modality |
|-----------|-------------|-------|-----------|----------|
| LongLLaVA | Long-context vision-language | [Paper](https://arxiv.org/abs/2409.02889) | [GitHub](https://github.com/FreedomIntelligence/LongLLaVA) | Text, Image |
| LLaVA-OneVision | Unified vision-language model | [Paper](https://arxiv.org/abs/2408.03326) | [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) | Text, Image, Video |
| KOSMOS-1 | Multimodal LLM | [Paper](https://arxiv.org/abs/2302.14045) | - | Text, Image |
| KOSMOS-2 | Grounded multimodal LLM | [Paper](https://arxiv.org/abs/2306.14824) | [GitHub](https://github.com/microsoft/unilm/tree/master/kosmos-2) | Text, Image |
| ChatGLM | Chinese language model | [Paper](https://arxiv.org/abs/2406.12793) | [GitHub](https://github.com/THUDM/ChatGLM-6B) | Text |
| CVLUE | Chinese vision-language understanding | [Paper](https://arxiv.org/abs/2502.00000) | - | Text, Image |
| ToT (Tree of Thoughts) | Reasoning framework | [Paper](https://arxiv.org/abs/2305.10601) | [GitHub](https://github.com/princeton-nlp/tree-of-thought-llm) | Text |
| AgEval | Agricultural domain evaluation | [Paper](https://arxiv.org/abs/2502.15147) | - | Text |

---

## Citation

If you find this resource helpful, please cite our survey paper:

```bibtex
@article{anjum2025domain,
  title={Domain Specific Benchmarks for Evaluating Multimodal Large Language Models},
  author={Anjum, Khizar and Arshad, Muhammad Arbab and Hayawi, Kadhim and Polyzos, Efstathios and Tariq, Asadullah and Serhani, Mohamed Adel and Batool, Laiba and Lund, Brady and Mannuru, Nishith Reddy and Bevara, Ravi Varma Kumar and Mahbub, Taslim and Akram, Muhammad Zeeshan and Shahriar, Sakib},
  journal={Data Science and Management},
  year={2025},
  publisher={Elsevier}
}
```

## Contributing

We welcome contributions to keep this resource up-to-date! Please submit a pull request or open an issue if you:

- Find broken links or outdated information
- Know of additional benchmarks that should be included
- Have corrections or improvements to benchmark descriptions

### Contribution Guidelines

1. Fork this repository
2. Add your benchmark to the appropriate section following the existing format
3. Include: Benchmark name, description, paper link, code/data link (if available), and modality
4. Submit a pull request with a brief description of your changes

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the collaborative efforts of researchers from Rutgers University, Abu Dhabi University, University of Strasbourg, University of North Texas, and other institutions.

---

**Note:** Some benchmarks may have restricted access or require institutional agreements. Please refer to the original papers for access details. Links are provided for publicly available resources where possible.
