# RoadSentinel
## An Agentic Vision System for Uncertainty-Aware Vulnerable Road User Analysis

**Competition:** OpenCV AI Competition 2026, powered by AWS  
**Primary dataset:** BDD100K — Berkeley DeepDrive  
**Core technologies:** OpenCV 5, pretrained YOLO, agentic orchestration, AWS, and Cloud-Optimized OpenCV Library (COOL)

---

## 1. Problem Statement

### 1.1 Background

Vulnerable road users—including pedestrians, cyclists, and motorcyclists—are an important focus of road-scene analysis because their visibility and movement can be difficult to interpret in complex traffic environments. They may be partially obscured by vehicles, appear briefly between frames, occupy relatively few pixels, or become harder to distinguish under challenging lighting and weather conditions.

Modern object-detection models can identify road users in individual images, but a detection is not necessarily a complete assessment of the surrounding situation. A missed detection, an unstable prediction, or an ambiguous observation may require additional visual evidence before a system can produce a well-supported conclusion.

Conventional single-pass pipelines generally apply a predefined sequence of operations. They do not necessarily decide whether a particular observation warrants further investigation, which additional evidence would be most useful, or whether further computation is justified.

Processing every available frame with the same intensive analysis can also introduce unnecessary computational overhead, particularly when consecutive frames contain redundant information.

### 1.2 Core Challenge

**How can an agentic vision system improve the reliability of vulnerable road user analysis by autonomously selecting and evaluating additional visual evidence when initial detections are uncertain, while maintaining measurable accuracy, computational efficiency, and transparent decision-making?**

The central challenge is to move beyond detecting objects toward evidence-driven visual investigation.

The system must determine when its initial analysis is insufficient, choose an appropriate follow-up operation, incorporate the resulting evidence into its assessment, and communicate the outcome with an auditable record of its decisions.

It must also demonstrate experimentally whether this adaptive investigation provides a meaningful advantage over conventional single-pass object detection.

---

## 2. Proposed Solution

RoadSentinel is an agentic road-scene analysis application that identifies vulnerable road users in real-world driving footage and selectively investigates uncertain or potentially significant observations.

The system will use BDD100K as its primary dataset and begin with a verified pretrained object detector, such as a compatible YOLO checkpoint. OpenCV 5 will perform substantive image and video analysis, including frame extraction, region-of-interest processing, image transformations, and visual comparison.

Following the initial detection pass, an orchestration agent will evaluate the available evidence and determine whether further investigation is warranted. Depending on the observation, it may:

- Inspect a different frame from the same driving sequence.
- Extract and analyze a crop around a pedestrian, cyclist, or rider.
- Compare detections across frames using temporal information.
- Invoke another suitable vision operation when the initial evidence is inconclusive.
- Escalate the observation for human review when the available evidence remains insufficient.

The agent will not automatically treat a low-confidence detection as a dangerous event. Instead, it will use the uncertainty and context of the observation to decide whether further analysis is justified.

The final output will include relevant detections, supporting frames, object-tracking information where available, the investigation history, and a structured assessment with explicit uncertainty disclosures.

RoadSentinel will be deployed using AWS. Its image-processing workload will also be evaluated using COOL on AWS Graviton to determine whether the optimized implementation provides measurable performance benefits over an appropriate baseline.

---

## 3. Target Users and Intended Impact

### Primary users

- Road-safety researchers studying vulnerable road users.
- Transportation analysts reviewing dashcam and traffic-camera footage.
- Computer-vision developers evaluating temporal perception and agentic investigation.
- Organizations that need to prioritize recorded footage for human inspection.

### Intended impact

RoadSentinel aims to help users identify and review difficult road-scene observations more systematically. Rather than requiring every frame to receive identical computational treatment, the system will investigate selected observations when additional evidence may improve the assessment.

The project will investigate whether this approach can improve the quality of visual assessments while controlling additional processing cost.

Any claimed improvement in detection quality, review efficiency, or computational cost will be established through experiments rather than assumed in advance.

---

## 4. Dataset and Experimental Design

### Primary dataset: BDD100K

BDD100K contains diverse real-world driving footage and supports multiple computer-vision tasks, including object detection and multi-object tracking.

The project will use the relevant dataset components:

- **Object-detection images and labels:** to establish a baseline for detecting pedestrians, riders, motorcycles, and bicycles.
- **Tracking images and annotations:** to evaluate object continuity and temporal analysis on the available annotated tracking sequences.
- **Original driving videos:** to provide additional temporal context where access and the selected evaluation protocol permit their use.

The project will begin with a representative subset rather than attempting to process the entire dataset immediately.

The evaluation will preserve training, validation, and test separation. When assessing temporal investigation, we will use annotated tracking sequences or a clearly documented evaluation subset with appropriate reference labels.

Where labels are unavailable for a specific claim, the project will not present that claim as quantitatively validated.

### Research question

Does an agent that selectively investigates uncertain vulnerable road user observations achieve better assessment quality, better computational efficiency, or a more favorable trade-off between the two than a conventional single-pass pipeline?

---

## 5. Agentic Vision Workflow

RoadSentinel will implement an explicit perception–decision–action loop.

**Stage 1 — Initial perception**

OpenCV 5 prepares the input, and the pretrained detector identifies supported object classes and returns bounding boxes and confidence scores.

**Stage 2 — Evidence assessment**

The system evaluates relevant signals, such as detection confidence, object size, occlusion indicators where available, consistency across frames, and tracking stability.

**Stage 3 — Action selection**

The agent decides whether to accept the current result, inspect another frame, analyze a targeted crop, invoke another vision operation, or request human review.

**Stage 4 — Follow-up analysis**

OpenCV 5 and the selected vision components execute the chosen action. The resulting evidence is compared with the initial observation.

**Stage 5 — Final assessment**

The system records the final decision, supporting evidence, remaining uncertainty, and the actions taken during the investigation.

The agent must be able to select different actions for different observations. A fixed pipeline that always performs the same operations does not demonstrate the intended adaptive behavior.

---

## 6. Technical Objectives

The project has five measurable objectives.

1. **Detection quality:** Establish and evaluate a reproducible baseline for the selected vulnerable road user classes.

2. **Temporal consistency:** Evaluate whether incorporating evidence across frames improves object-level assessment compared with isolated-frame analysis.

3. **Agentic effectiveness:** Determine whether adaptive follow-up actions improve final assessment quality over single-pass inference.

4. **Computational efficiency:** Measure the processing overhead introduced by agentic investigation and investigate whether selective frame analysis reduces unnecessary work.

5. **Cloud optimization:** Benchmark relevant OpenCV operations using COOL on AWS Graviton and report reproducible performance comparisons.

These objectives are independent: an agent may improve assessment quality while increasing latency, and an optimized image-processing pipeline may improve throughput without improving detection accuracy. Both benefits and trade-offs must be reported honestly.

---

## 7. Evaluation Framework

| Dimension | Evaluation method |
|---|---|
| Object detection | mAP@50, mAP@50–95, precision, recall, and class-wise results |
| Temporal analysis | Tracking metrics and consistency of observations across annotated sequences |
| Agentic effectiveness | Final assessment quality, follow-up action success, and comparison with a single-pass baseline |
| Investigation efficiency | Number of frames and additional operations processed per assessment |
| Runtime performance | End-to-end latency, throughput, resource utilization, and processing cost |
| COOL performance | Reproducible baseline-versus-COOL comparison on documented AWS Graviton hardware |
| Reliability | Failure cases, false detections, missed objects, unresolved uncertainty, and appropriate human escalation |

The experiments will use fixed evaluation inputs and clearly documented configurations. Where possible, results will include repeated measurements and variability rather than relying on a single favorable run.

### Primary success criterion

The principal research objective is to establish whether adaptive visual investigation improves the quality of vulnerable road user assessments over single-pass inference, and to quantify the additional computational cost or savings associated with that improvement.

The project will not define arbitrary accuracy gains in advance. Numerical targets will be established after the baseline is measured and the evaluation protocol is finalized.

---

## 8. Competition Alignment

### Main competition

RoadSentinel will deliver a functioning computer-vision application that uses OpenCV 5 for substantive image or video analysis and runs a meaningful component on AWS.

### Agentic Vision Award

The system will demonstrate that initial visual evidence changes the agent's subsequent action. A recorded investigation trace will show the observation, the decision, the follow-up tool call or action, and the resulting assessment.

### Best Use of COOL Award

The project will run a claimed core image-processing workload using COOL on AWS Graviton. It will document the implementation, hardware configuration, inputs, baseline, and reproducible performance results.

The three evaluation objectives will remain distinct. A successful detector alone will not establish agentic behavior, and the use of COOL alone will not establish a performance advantage.

---

## 9. Scope and Limitations

RoadSentinel is a road-scene analysis and research prototype, not a certified autonomous-driving or collision-prediction system.

The initial scope is limited to supported vulnerable road user classes and evidence that can be established from the available data. The system will not claim to detect every road hazard, predict every collision, or determine real-world risk with certainty.

BDD100K represents particular geographic and environmental conditions; results on that dataset will not establish generalization to all roads, countries, or camera systems.

Human review will remain available when observations cannot be resolved reliably. The system will preserve uncertainty rather than force a definitive classification when evidence is insufficient.

---

## 10. Final Vision

RoadSentinel seeks to answer a more useful question than simply identifying objects in a driving video:

**When the visual evidence is uncertain, what should the system examine next, and does that additional investigation produce a better-supported assessment at an acceptable computational cost?**

By combining pretrained object detection, substantive OpenCV 5 analysis, adaptive agentic investigation, real-world driving data, and reproducible cloud optimization, RoadSentinel will test a practical approach to building more reliable and efficient vision systems.

Its contribution will be demonstrated through a working application, transparent agent traces, controlled experiments, documented failure cases, and measured performance—not merely through the integration of existing tools.
