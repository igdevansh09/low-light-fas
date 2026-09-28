# MID-SEM PROGRESS REPORT

## Low-Light Face Anti-Spoofing and Identity Verification

**Submitted by:**

| Student | Roll number |
| --- | --- |
| Priyanshu Yadav | 2023UEA3643 |
| Saksham | 2023UEA6654 |
| Harshit | 2023UEA6601 |
| Devansh Gupta | 2023UEA6605 |

**Department:** Electronics and Communication Engineering  
**Institution:** NSUT  
**Supervisor:** Rashmi Gupta  
**Academic Session:** 2026-27  
**Date:** _________________________

**Supervisor's Signature:** __________________________________

---

### Report Status

| Item                           | Status                                                    |
| ------------------------------ | --------------------------------------------------------- |
| Application prototype          | Implemented                                               |
| Anti-spoofing integration      | Implemented with two local MiniFASNet weights             |
| Identity verification pipeline | Implemented with UniFace detection and ArcFace embeddings |

### Contents

1. [Introduction](#1-introduction)
2. [Literature Survey](#2-literature-survey)
3. [Problem Statement and Methodology](#3-problem-statement-and-methodology)
4. [Work Done Till Date](#4-work-done-till-date)
5. [Results and Discussion](#5-results-and-discussion)
6. [Future Work](#6-future-work)
7. [Conclusion](#7-conclusion)
8. [References](#references)
9. [Appendices](#appendix-a-current-project-file-mapping)

---

## Abstract

Face recognition can be unreliable when the captured face is poorly illuminated, and it can also be deceived by a photograph, replayed display, or other presentation attack. This project develops a prototype that combines face anti-spoofing with reference-based identity verification. The current application accepts a trusted reference image and a submitted target image through a Streamlit interface. It first evaluates the target with two local MiniFASNet anti-spoofing models. If the target passes the liveness stage, the application detects faces and facial landmarks, extracts normalized ArcFace embeddings, and compares the reference and target embeddings using cosine similarity.

The work completed so far includes investigation of suitable anti-spoofing approaches, assessment of available pretrained models, integration of the selected models, assembly of the complete verification pipeline, and development of an interactive interface. The current implementation is a functional image-upload prototype. It reports liveness confidence, face-detection status, similarity, and the final verification decision. It does not yet provide a controlled low-light benchmark, a calibrated operating threshold, or a continuous camera workflow. Therefore, this report distinguishes implemented functionality from results that still require experimental measurement.

**Keywords:** face anti-spoofing, presentation attack detection, low-light imaging, MiniFASNet, RetinaFace, ArcFace, cosine similarity, biometric verification.

---

## 1. Introduction

### 1.1 Background

Biometric verification systems compare a submitted sample with a previously enrolled reference. In face verification, the system must answer two different questions. First, does the submitted sample represent a live person rather than an attack medium? Second, does the live face belong to the enrolled identity? Treating these questions as one task can expose the recognition stage to presentation attacks. A photograph or screen replay may contain enough visual information to produce a plausible identity match even though the real person is not present.

Lighting adds another source of uncertainty. In low illumination, the face region may have weak contrast, increased sensor noise, colour distortion, shadows, and missing facial detail. These effects can reduce the reliability of face detection, facial landmark localisation, liveness classification, and identity matching. A practical system must therefore expose intermediate decisions and make it possible to determine which stage failed.

The present project addresses this problem by combining a lightweight anti-spoofing subsystem with a face detection and recognition subsystem. The prototype is intentionally modular. Anti-spoofing is performed before identity matching, and the identity decision is made only after the submitted face passes the liveness check.

### 1.2 Motivation

Many face-verification demonstrations show only a recognition score. Such a score does not establish that the input came from a live subject. The motivation of this project is to build a more defensible verification flow in which a submitted image is first checked for presentation-attack risk and is then compared with a trusted reference.

The low-light setting is important because the same pipeline may behave differently under normal and dim illumination. A model that performs well on clear, well-lit images may lose confidence when the face is dark or partially obscured by noise. The project therefore treats low-light robustness as an evaluation objective. The current stage establishes the software pipeline required for that evaluation; it does not yet claim a measured improvement in low-light accuracy.

### 1.3 Aim

The aim is to develop an image-based biometric verification prototype that combines presentation-attack detection and reference-based face matching, with particular attention to operation in challenging illumination.

### 1.4 Objectives

1. Study face anti-spoofing methods and select an implementation compatible with the project requirements.
2. Integrate pretrained MiniFASNet anti-spoofing weights into a usable inference pipeline.
3. Detect faces and facial landmarks in reference and target images.
4. Extract normalized face embeddings and compare them using cosine similarity.
5. Build an interactive interface that exposes intermediate scores and final decisions.
6. Identify the current limitations of the prototype and define a repeatable low-light evaluation protocol.
7. Determine whether low-light enhancement, threshold calibration, or model adaptation is justified by measured results.

### 1.5 Scope

The present implementation supports JPG, JPEG, and PNG image uploads. One image is used as a trusted reference and another as the submitted target. The application runs the anti-spoofing and recognition stages in sequence and displays the outcome in a Streamlit interface.

The present scope does not include continuous video capture, mobile deployment, a production database, automatic enrolment management, or a completed statistical benchmark. The included training and testing scripts provide a basis for future experimentation, but the running application currently uses the available pretrained weights.

---

## 2. Literature Survey

### 2.1 Face recognition and verification

Face recognition systems commonly transform a detected face into a compact numerical representation called an embedding. Images from the same identity are expected to produce embeddings that are closer to each other than embeddings from different identities. A verification system compares two embeddings and applies a decision rule. Cosine similarity is a common choice because it measures the angular agreement between two vectors and is less affected by their magnitude when embeddings are normalized.

A recognition score alone is not a liveness guarantee. The input can be a picture of a genuine person while still being an attack sample. This motivates a separate presentation-attack detection stage.

### 2.2 Face detection and landmark localisation

A detector identifies the face region and, in many systems, returns facial landmarks. Reliable landmarks are useful for aligning the eyes, nose, and mouth before feature extraction. The current project uses the UniFace interface with a RetinaFace-based detector. The detector is used for both the trusted reference and the submitted target. The system proceeds to identity embedding extraction only when usable faces are detected in both images.

The anti-spoofing component has its own face-region detection path. This is separate from the recognition detector because the anti-spoofing model expects a model-specific crop. Maintaining these roles separately makes the current data flow explicit and allows either subsystem to be evaluated independently in future work.

### 2.3 Presentation-attack detection

Presentation-attack detection attempts to distinguish a bona fide live face from an artificial presentation. Attacks may include printed photographs, images displayed on a phone or monitor, replayed video, masks, or other substitutes. A useful detector should preserve sensitivity to these attacks while avoiding excessive rejection of genuine faces.

The project uses the MiniFASNet model family. These networks are compact and suitable for face anti-spoofing inference. The repository contains two local model files: a MiniFASNetV2 weight and a MiniFASNetV1SE weight. The application runs both model files, accumulates their three-class prediction outputs, selects the highest-scoring class, and converts the selected accumulated value into a displayed liveness score. In the current decision logic, class `1` is treated as the live-face class.

### 2.4 Low-light face analysis

Low-light face analysis can be improved through image enhancement, camera-level exposure control, denoising, illumination-invariant representations, or training with varied lighting. However, enhancement may also amplify noise or introduce artificial patterns that affect liveness decisions. A method should therefore be judged experimentally rather than assumed to improve performance.

For this project, the first priority is to establish a baseline using the current pipeline. Samples should then be collected under normal, dim, and very dim conditions. The same identities and attack types should be represented across conditions, while development and test captures should remain separate. This will show whether failures originate primarily in anti-spoofing, face detection, embedding extraction, or the final similarity threshold.

### 2.5 Research gap addressed by the project

The project focuses on integration and evaluation readiness. Many prototype demonstrations stop after recognition, while this work combines liveness and identity decisions in one workflow and reports the intermediate telemetry. The remaining research gap is quantitative: the current repository does not yet establish how the complete pipeline behaves across illumination levels or presentation-attack types. The next stage must supply this evidence.

---

## 3. Problem Statement and Methodology

### 3.1 Problem statement

A face verification system operating in dim light must reject presentation attacks and verify the identity of a live subject. It should also provide an interpretable response when the face cannot be detected, when the liveness classifier rejects the input, or when the face is live but does not match the reference.

The problem can be expressed as a two-stage decision:

1. Determine whether the target image is a bona fide live-face sample.
2. If it is live, determine whether its identity embedding is sufficiently similar to the reference embedding.

The system must not claim successful identity verification when the liveness stage fails.

### 3.2 System architecture

The implemented architecture contains the following modules:

- **Input module:** accepts a trusted reference image and a target image.
- **Anti-spoofing module:** detects a face region, generates the required crop, and evaluates the crop using the available MiniFASNet weights.
- **Recognition detection module:** detects faces and landmarks in both images through UniFace.
- **Embedding module:** produces normalized ArcFace embeddings from the detected facial landmarks.
- **Similarity module:** computes cosine similarity between the two embeddings.
- **Decision module:** combines liveness status, detection status, embedding status, and similarity threshold.
- **Interface module:** displays images, scores, pipeline status, diagnostics, and the final result.

#### Figure 1. Simple view of the implemented pipeline

**Reference + target images** → **Anti-spoofing check** → **Face detection and matching** → **Verification result**

The architecture separates liveness from identity matching. This ordering is central to the security logic: a target that fails liveness cannot proceed to an identity-verified result.

### 3.3 Detailed processing flow

#### Step 1: Reference enrolment

The operator uploads a reference image in the sidebar. The image is decoded with OpenCV and retained as the trusted identity image. The current interface recommends a clear, frontal reference image because poor enrolment quality can affect the later comparison.

#### Step 2: Target image input

The operator uploads a target frame. The target is decoded from its byte stream into an OpenCV image. If decoding fails, the pipeline stops and reports the input error.

#### Step 3: Anti-spoofing face region

The anti-spoofing detector searches the target image for a face region. The cropper uses the detected bounding box and the scale encoded in each model filename. Crops are resized to the input dimensions expected by the corresponding model. The available weights use 80 by 80 input patches.

#### Step 4: MiniFASNet ensemble inference

Each local weight is loaded on the available CPU or CUDA device and evaluated on the prepared crop. The prediction vectors are accumulated. The class with the largest accumulated value is selected. The application treats class `1` as a live-face result and rejects other classes as presentation attacks. This ensemble is an integration of the available weights; it is not a newly trained ensemble benchmark.

#### Step 5: Face detection and landmarks

If the target passes the liveness condition, the system detects faces in both the reference and target images. The current implementation uses the first detected face from each result. A missing face in either image produces a face-alignment failure rather than an identity decision.

#### Step 6: Embedding extraction

The recognizer uses the detected landmarks to produce normalized ArcFace embeddings. Extraction is performed independently for the reference and target images. Errors in this stage are reported as embedding extraction failures.

#### Step 7: Similarity and identity decision

The system computes cosine similarity between the normalized embeddings. The current prototype uses a fixed similarity threshold of `0.40`. A score at or above this threshold produces an identity-verified result after liveness has passed. A lower score produces an identity mismatch. This threshold is a prototype setting and requires calibration on representative development data before operational use.

### 3.4 Software and hardware requirements

The application is implemented in Python. Streamlit provides the web interface, OpenCV handles image decoding and image operations, PyTorch loads the anti-spoofing models, and UniFace provides the detector and recognizer interface. The repository also contains the original anti-spoofing training dependencies, including TensorBoard logging and dataset-loader utilities.

The anti-spoofing inference path supports CPU execution and uses CUDA when available. The model loader maps weights to the selected device. The current application is therefore usable without requiring a dedicated GPU, although inference speed may differ substantially between CPU and GPU execution.

### 3.5 Decision states shown by the interface

The interface separates the following cases:

- Face not detected by the anti-spoofing subsystem.
- Presentation attack rejected by the liveness classifier.
- Face alignment failure in the reference or target image.
- Embedding extraction failure.
- Live subject detected with identity verified.
- Live subject detected with identity mismatch.

This separation is important for later error analysis because a failure to verify may arise from different causes.

---

## 4. Work Done Till Date

### 4.1 Requirement analysis and research

The team studied the requirements of a verification system that must consider both liveness and identity. The initial research compared the role of anti-spoofing models, face detectors, and recognition models. The project direction was refined from a general face-recognition idea into a staged face anti-spoofing and identity-verification pipeline suitable for low-light evaluation.

The anti-spoofing model search focused on an implementation that could be integrated with the existing project structure and run locally. The selected model family is compatible with the available inference code, crop generation logic, model-name parsing, and PyTorch loading path.

### 4.2 Model selection and repository integration

The repository was organised around the MiniFASNet anti-spoofing model family. Two model weights are available under the local anti-spoofing resource directory. The associated model definitions, crop generation, prediction code, and configuration utilities are present in the project.

The current application loads both available weights during target evaluation. The prediction outputs are combined before the liveness class is selected. This allows the application to use the resources already available in the repository rather than requiring a new model-download step during every run.

### 4.3 Training investigation

The repository includes a training path based on a MultiFTNet model, a dataset loader, data augmentation, classification loss, feature loss, stochastic gradient descent, learning-rate milestones, TensorBoard logging, and periodic model checkpoints. Its configuration points to `datasets/rgb_image` (with a patch-size subdirectory), but the repository does not include a dataset manifest or identify a named dataset used in the team's training attempt.

An in-house model-training attempt was made, but the available project records do not identify its dataset or establish that its checkpoint is used by the application. The current running application uses the bundled pretrained MiniFASNetV2 and MiniFASNetV1SE weights. Therefore, no specific dataset can be accurately credited as the training dataset for the model currently used by the application, and no training accuracy or project-trained checkpoint is claimed here. A reproducible training experiment can be added when a verified, labelled, and appropriately separated dataset is documented.

**Dataset used for the running model:** The repository does not specify the original training dataset for the bundled pretrained MiniFASNet weights. The training configuration only names the expected local directory `datasets/rgb_image`; this is a path, not a dataset name. Accordingly, the exact dataset used for the team's training attempt is **not documented in the current project files**.

### 4.4 Pipeline assembly

The complete application pipeline was assembled in the main Streamlit application. The implementation includes model loading, image upload, anti-spoofing inference, face detection, landmark-based embedding extraction, cosine similarity, threshold comparison, and user-facing result states.

The application also includes a telemetry view. It reports the liveness score, whether the reference and target faces were detected, whether embeddings were generated, the similarity value, and the configured threshold. This makes the prototype useful for debugging and for collecting evidence during the next evaluation phase.

### 4.5 User interface development

The interface was developed as a verification workspace rather than as a static demonstration. It provides separate areas for the trusted reference and submitted frame, a visible pipeline sequence, a security decision panel, score cards, and a technical telemetry section.

The current interface supports the following operational flow:

1. Upload a reference image.
2. Upload a target image.
3. Wait for local model loading and pipeline execution.
4. Inspect the liveness and similarity information.
5. Read the final verification, mismatch, or rejection state.

### 4.6 Current team work represented without individual attribution

The completed work can be summarised as four coordinated activities: anti-spoofing model research and selection, research into suitable pretrained resources, implementation of the model-training path, and assembly of the end-to-end application pipeline. The final running prototype combines these activities into one reproducible workflow.

---

## 5. Results and Discussion

### 5.1 Implemented results

The current project has reached the stage of a functional integration prototype. It can accept a reference image and target image, execute the anti-spoofing stage, and conditionally continue to face recognition. It displays a liveness score when a usable face region is found and reports a separate rejection state when the liveness class is not live.

When liveness passes, the system attempts face detection and landmark extraction for both images. If both faces are available, it generates normalized embeddings and computes cosine similarity. The application can then report either identity verification or identity mismatch according to the configured threshold.

### 5.2 Observed decision logic

The implementation enforces the intended security ordering. The identity embedding stage is entered only when the target has been classified as live and both reference and target faces have been detected. This prevents a target rejected as a presentation attack from being presented as a successful identity match.

The separate failure messages also make it possible to distinguish several practical causes of rejection. A dark target may fail at anti-spoofing, face detection, or embedding extraction; a bright but unrelated face may pass liveness and be rejected only at identity matching. The current interface exposes enough intermediate information to record these cases during testing.

### 5.3 Results not yet established

No numerical accuracy, APCER, BPCER, ACER, FAR, FRR, EER, or low-light improvement value is claimed in this report. The repository does not currently contain a completed held-out benchmark with documented lighting conditions, attack categories, sample counts, and threshold calibration.

The liveness score displayed by the interface is a model confidence-style output from the accumulated predictions. It should not be described as a validated probability until calibration and evaluation are performed. Similarly, the `0.40` identity threshold is a current application setting, not a scientifically selected operating point.

### 5.4 Proposed evaluation protocol

The next evaluation should contain three broad sample groups:

- Bona fide samples from enrolled subjects.
- Impostor samples from non-matching subjects.
- Presentation attacks such as printed photographs, screen displays, and replayed content where available.

Each group should be evaluated under normal, dim, and very dim illumination. The development set should be used to calibrate the similarity threshold and inspect liveness behaviour. The final test set should remain held out until the settings are fixed. Identities and capture sessions should be separated between development and test sets to reduce leakage.

For presentation-attack detection, report APCER, BPCER, and ACER. For identity verification, report FAR, FRR, and EER where the data support those measurements. Also record face-detection failures, embedding failures, processing time, and results broken down by illumination condition.

#### Table 1. Proposed evaluation matrix

| Evaluation dimension | Conditions or classes                    | Primary observations                   |
| -------------------- | ---------------------------------------- | -------------------------------------- |
| Subject type         | Bona fide, impostor, presentation attack | Liveness and identity decisions        |
| Illumination         | Normal, dim, very dim                    | Detection failures and score stability |
| Anti-spoofing        | Live class versus attack rejection       | APCER, BPCER, ACER                     |
| Identity matching    | Same identity versus different identity  | FAR, FRR, EER                          |
| Runtime              | CPU and CUDA, where available            | Model-loading and per-image latency    |

### 5.5 Discussion of limitations

The current prototype uses a single selected face from each detection result. Multiple-face scenes are not yet handled as a separate workflow. The model weights are used as supplied, and their original training conditions may not represent the target low-light environment. The current system also processes uploaded still images rather than a continuous camera stream.

The application does not currently include explicit low-light enhancement. This is a deliberate limitation in the baseline stage because enhancement should be compared experimentally against the unprocessed input. Adding a preprocessing method without a controlled comparison could make the results difficult to interpret.

Privacy and security requirements have also not been developed into a deployment policy. A production system would need consent, secure handling of biometric data, retention limits, access control, and protection against misuse.

---

## 6. Future Work

1. **Create a controlled dataset:** Collect genuine, impostor, and presentation-attack samples at multiple illumination levels while recording capture conditions.
2. **Calibrate thresholds:** Select the liveness and similarity operating points using a development set rather than relying on the current fixed value.
3. **Complete quantitative evaluation:** Report APCER, BPCER, ACER, FAR, FRR, EER, detection failures, and latency.
4. **Study low-light preprocessing:** Compare the baseline with carefully selected enhancement, denoising, or exposure-normalisation methods.
5. **Evaluate model adaptation:** Train or fine-tune an anti-spoofing model only with a verified dataset and compare it against the supplied pretrained weights.
6. **Improve multi-face handling:** Detect and report multiple faces instead of silently using only the first result.
7. **Add camera or video input:** Extend the image-upload workflow to controlled video frames after still-image performance is understood.
8. **Improve reliability:** Add explicit handling for invalid bounding boxes, model-loading errors, unsupported image conditions, and device availability.
9. **Measure deployment performance:** Record CPU and GPU inference time, memory use, and model-loading overhead.
10. **Address biometric privacy:** Define consent, storage, deletion, access, and audit procedures before any real-world deployment.

---

## 7. Conclusion

The project has progressed from model research and resource selection to a functional face anti-spoofing and identity-verification prototype. The application now assembles the complete staged workflow: target face-region detection, MiniFASNet liveness inference, UniFace face detection, ArcFace embedding extraction, cosine similarity, and final identity decision.

The most important result at this stage is the working integration and its diagnostic interface. The system can distinguish liveness rejection, face-detection failure, embedding failure, identity mismatch, and identity verification. This provides a practical foundation for the next phase of controlled testing.

The project should not yet be presented as a validated low-light biometric system because measured benchmark results, threshold calibration, and a controlled illumination study are still pending. The next milestone is to establish a documented dataset and evaluation protocol, quantify the baseline, and then test whether low-light-specific processing or model adaptation provides a measurable benefit.

---

## References

1. MiniVision, _Silent-Face-Anti-Spoofing_, MiniFASNet implementation and model family documentation.
2. Deng, J. et al., _RetinaFace: Single-stage Dense Face Localisation in the Wild_, arXiv preprint, 2019.
3. Deng, J. et al., _ArcFace: Additive Angular Margin Loss for Deep Face Recognition_, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.
4. Wang, Y. et al., _Face Anti-Spoofing Using a Multi-Channel Convolutional Neural Network_, research literature on presentation-attack detection.
5. ISO/IEC 30107-3, _Information technology: Biometric presentation attack detection — Testing and reporting_.
6. The project repository source files: Streamlit application, anti-spoofing inference code, MiniFASNet model definitions, crop-generation utilities, training configuration, and dataset loader.

---

## Appendix A: Current Project File Mapping

| Project component       | Current implementation                                          |
| ----------------------- | --------------------------------------------------------------- |
| Interface               | Streamlit application in `app.py`                               |
| Image decoding          | OpenCV                                                          |
| Anti-spoofing detector  | OpenCV DNN face-region detector                                 |
| Anti-spoofing models    | MiniFASNetV2 and MiniFASNetV1SE local weights                   |
| Anti-spoofing crop      | Model-specific crop and resize utility                          |
| Face detection          | UniFace RetinaFace interface                                    |
| Identity recognition    | UniFace ArcFace interface                                       |
| Similarity              | Cosine similarity                                               |
| Current match threshold | `0.40`                                                          |
| Supported input         | JPG, JPEG, PNG uploads                                          |
| Current output          | Liveness, detection, embedding, similarity, and decision states |

## Appendix B: Suggested Demonstration Sequence

1. Start the Streamlit application.
2. Upload a clear reference image.
3. Upload a genuine target image in normal illumination.
4. Record liveness score, detection status, similarity, and final decision.
5. Repeat with a non-matching subject.
6. Repeat with a presentation-attack sample.
7. Repeat the three cases under dim and very dim illumination.
8. Store the observations in a results table without changing thresholds between test cases.
9. Use development observations for calibration and reserve final samples for held-out reporting.
