# Project Synopsis

## Low-Light Face Anti-Spoofing

**Student(s):** Devansh Gupta, Saksham Chaudhary, Harshit, Priyanshu Yadav  
**Department:** Electronics and Communication Engineering  
**Institution:** NSUT  
**Supervisor:** Rashmi Gupta

### Supervisor's Approval

**Supervisor:** Rashmi Gupta  
**Signature:**

<br><br><br>

---

**Date:** **********\_\_**********

### Abstract

This project presents a prototype biometric verification system that combines face anti-spoofing with face identity matching in a dim light. A user supplies a trusted reference image and a target image through a Streamlit interface. The system first evaluates whether the target appears to be a live face using two included MiniFASNet model weights. If the liveness stage passes, it detects and aligns faces, extracts ArcFace embeddings, and compares them using cosine similarity. The interface reports liveness and similarity scores together with a verification, mismatch, or rejection outcome.

The project is motivated by the difficulty of dependable face verification in challenging capture conditions, including low illumination. The current implementation provides an integration prototype for the anti-spoofing and identity-matching pipeline; low-light-specific enhancement, controlled low-light testing, and benchmark results remain work to be completed. This distinction makes low-light robustness a testable research objective rather than an established result.

### Problem Statement

Face recognition systems may accept a photograph, display, or other presentation attack as a genuine person, while changes in illumination can make face detection and identity matching less reliable. A verification workflow therefore needs to assess both whether the submitted face appears live and whether it matches the enrolled reference. It also needs to be evaluated under varied lighting before claims about low-light performance can be made.

### Aim and Objectives

The aim is to develop and evaluate an image-based face verification prototype that combines presentation-attack detection with reference-based identity matching, with particular attention to its behavior under low-light conditions.

The objectives are to:

1. Integrate a face anti-spoofing stage with a face detection and recognition pipeline.
2. Compare a submitted face against a user-provided reference using facial embeddings and cosine similarity.
3. Present intermediate scores and the final decision in an interactive interface.
4. Establish a repeatable evaluation protocol across lighting conditions and genuine, impostor, and presentation-attack samples.
5. Identify failure cases and determine whether illumination-specific preprocessing or model adaptation is warranted.

### Methodology and System Design

The prototype is implemented in Python. Streamlit provides the user interface, while OpenCV handles image decoding and face-region cropping. PyTorch loads the included MiniFASNet anti-spoofing weights. UniFace supplies a RetinaFace-based detector and an ArcFace recognizer.

The current verification workflow is:

1. The operator uploads a trusted reference image and a target image.
2. The anti-spoofing component detects a face region, prepares model-specific crops, and runs the available MiniFASNet weights. Their predictions are combined to produce a liveness class and score.
3. UniFace detects faces and facial landmarks in both images.
4. If the target passes the liveness check and faces are detected in both images, ArcFace produces normalized embeddings.
5. The system computes cosine similarity and compares it with the prototype's configured threshold of 0.40. It reports an identity match when the threshold is met; otherwise, it reports a mismatch. Spoof predictions and detection or embedding failures are rejected with separate messages.

The anti-spoofing implementation uses its own face-region detector for cropping. The identity stage uses UniFace detection and landmarks for embedding extraction. Keeping these roles distinct reflects the current code structure.

### Scope and Current Implementation

- The application supports uploaded JPG, JPEG, and PNG images; the current interface does not implement a continuous video or live-camera verification workflow.
- Two local anti-spoofing model weights are included: MiniFASNetV2 and MiniFASNetV1SE.
- The recognition decision uses a fixed similarity threshold in the application. It should be calibrated against representative validation data before any operational use.
- The repository includes training and testing scripts for the anti-spoofing model family, but the available project files do not establish that these included weights were trained specifically for this project.
- Low-light robustness is a motivation and evaluation target. The inspected application does not currently implement a dedicated low-light enhancement method or provide measured low-light performance.

### Proposed Evaluation

Evaluate the system on separate development and test data containing genuine users, non-matching identities, and presentation attacks. Capture or select samples under normal, dim, and very dim illumination, while keeping identities and capture sessions separated where feasible to reduce data leakage. Tune the decision thresholds on development data and report final results only on held-out test data.

For face anti-spoofing, report Attack Presentation Classification Error Rate (APCER), Bona Fide Presentation Classification Error Rate (BPCER), and Average Classification Error Rate (ACER). For identity verification, report False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER), along with detection failures and results broken down by illumination condition. Compare the current pipeline with and without any proposed low-light preprocessing to establish whether it improves performance and whether it introduces trade-offs.

These are proposed measurements; this repository does not currently provide a completed benchmark or numerical results.

### Expected Outcomes

The project is expected to deliver an interactive verification prototype, a documented evaluation protocol, and an empirical account of how lighting affects its liveness and identity decisions. The evaluation should identify conditions where the current models fail and provide evidence for or against adding low-light-specific preprocessing or model adaptation.

### Limitations and Future Work

The current prototype relies on pretrained model weights and a fixed similarity threshold, and no accuracy, error-rate, or low-light benchmark results are included. Performance may vary with camera quality, pose, occlusion, demographic representation, and presentation-attack type. The image-upload workflow also does not establish real-time performance or production suitability.

Future work should calibrate thresholds using representative data; evaluate on held-out, varied-lighting samples; add explicit low-light processing only if experiments support it; test additional attack types; and assess latency, privacy, and deployment security. Any deployment involving biometric data would also require appropriate consent, access controls, retention policies, and applicable institutional or legal review.
