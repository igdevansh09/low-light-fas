import os
import sys
import warnings

import cv2
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# UNI-FACE + ANTI-SPOOFING ONLY
# ============================================================

from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface import compute_similarity


# ============================================================
# ANTI-SPOOFING MODEL
# ============================================================

sys.path.append(
    os.path.abspath("antispoof_model")
)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Low-Light FAS | Identity Verification",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.html(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #050b12 0%, #0a1219 100%);
        color: #edf6ff;
    }

    .main .block-container {
        max-width: 1480px;
        padding-top: 24px;
        padding-bottom: 48px;
    }

    [data-testid="stSidebar"] {
        background: rgba(9, 15, 21, 0.96);
        border-right: 1px solid rgba(145, 168, 183, 0.12);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 18px;
    }

    .element-container {
        margin-bottom: 0.25rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 10px;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #7dd3fc 0%, #5eead4 100%);
        color: #06131c;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(94, 234, 212, 0.18);
    }

    .brand-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #edf6ff;
        line-height: 1.1;
    }

    .brand-subtitle {
        margin-left: 56px;
        color: #8da2b3;
        font-size: 13px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .system-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 18px 0 22px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(12, 28, 34, 0.9);
        border: 1px solid rgba(94, 234, 212, 0.2);
        color: #bfd4df;
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #5eead4;
        box-shadow: 0 0 12px rgba(94, 234, 212, 0.9);
    }

    .section-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 18px 0 12px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 800;
        color: #eef7ff;
    }

    .section-caption {
        color: #7d8ea0;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .panel {
        background: rgba(13, 22, 30, 0.96);
        border: 1px solid rgba(140, 170, 190, 0.18);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }

    .panel-meta {
        color: #7d97aa;
        font-size: 11px;
        letter-spacing: 0.14em;
        margin-top: 10px;
        text-transform: uppercase;
    }

    .st-key-trusted_identity_view,
    .st-key-submitted_frame_view {
        background: rgba(13, 22, 30, 0.96);
        border-color: rgba(140, 170, 190, 0.18) !important;
        border-radius: 16px !important;
        padding: 14px !important;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }

    .st-key-trusted_identity_view [data-testid="stImage"] img,
    .st-key-submitted_frame_view [data-testid="stImage"] img {
        border-radius: 10px;
    }

    .image-label {
        color: #90a7b4;
        font-size: 11px;
        letter-spacing: 0.16em;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .decision-panel {
        min-height: 240px;
        border-radius: 20px;
        padding: 22px 20px;
        background: linear-gradient(180deg, rgba(10, 16, 22, 0.96) 0%, rgba(9, 13, 18, 0.96) 100%);
        border: 1px solid rgba(146, 174, 196, 0.18);
        box-shadow: 0 18px 28px rgba(0, 0, 0, 0.16);
    }

    .decision-kicker {
        color: #7a90a0;
        font-size: 11px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .decision-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 10px;
    }

    .decision-copy {
        color: #a0b5c1;
        font-size: 13px;
        line-height: 1.6;
    }

    .decision-live { color: #6ee7b7; }
    .decision-danger { color: #ff7f7f; }
    .decision-neutral { color: #dfeaf3; }

    .score-card {
        background: rgba(11, 19, 25, 0.96);
        border: 1px solid rgba(141, 170, 190, 0.18);
        border-radius: 16px;
        padding: 16px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .score-label {
        color: #89a0af;
        font-size: 10px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .score-value {
        color: #eef7ff;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .score-detail {
        color: #7f96a7;
        font-size: 11px;
        margin-top: 6px;
    }

    .verified-box,
    .mismatch-box,
    .spoof-box,
    .warning-box {
        margin-top: 16px;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .verified-box {
        background: rgba(12, 34, 28, 0.9);
        border-color: rgba(89, 214, 168, 0.35);
    }

    .mismatch-box,
    .spoof-box {
        background: rgba(35, 17, 19, 0.9);
        border-color: rgba(255, 127, 127, 0.35);
    }

    .warning-box {
        background: rgba(30, 24, 13, 0.9);
        border-color: rgba(255, 206, 94, 0.35);
    }

    .pipeline {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 0 18px;
    }

    .pipeline-node {
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(14, 25, 33, 0.94);
        border: 1px solid rgba(146, 174, 196, 0.15);
        color: #bfd1db;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .pipeline-arrow {
        color: #5b7180;
        font-size: 12px;
        font-weight: 700;
    }

    .sidebar-brand {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #edf7ff;
        margin-bottom: 6px;
    }

    .sidebar-caption {
        color: #8ea4b3;
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .model-group {
        margin: 12px 0;
        padding: 12px;
        border: 1px solid rgba(142, 170, 188, 0.15);
        border-radius: 12px;
        background: rgba(11, 19, 25, 0.72);
    }

    .model-group-title {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        color: #e4edf4;
        font-size: 12px;
        font-weight: 700;
    }

    .model-group-note {
        margin-top: 3px;
        color: #718695;
        font-size: 10px;
    }

    .model-item {
        margin-top: 10px;
        padding-top: 9px;
        border-top: 1px solid rgba(142, 170, 188, 0.1);
        color: #b8cbd6;
        font-size: 11px;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .model-item strong {
        display: block;
        color: #dce8ee;
        font-size: 11px;
    }

    .model-item span {
        color: #718695;
        font-size: 10px;
    }

    .side-section {
        color: #aebfd1;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(11, 18, 24, 0.9);
        border: 1px solid rgba(133, 163, 181, 0.22);
        border-radius: 16px;
        padding: 8px;
    }

    [data-testid="stMetric"] {
        background: rgba(11, 19, 25, 0.9);
        border: 1px solid rgba(142, 170, 188, 0.18);
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(142, 170, 188, 0.18);
        border-radius: 16px;
        background: rgba(11, 19, 25, 0.9);
    }

    .stButton button {
        border-radius: 10px;
        border: 1px solid rgba(125, 211, 252, 0.2);
        background: linear-gradient(135deg, #0f1f2b 0%, #183748 100%);
        color: #ebfafe;
        font-weight: 700;
    }

    .stButton button:hover {
        border-color: rgba(94, 234, 212, 0.4);
        box-shadow: 0 8px 18px rgba(30, 123, 143, 0.18);
    }
    </style>
    """)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="brand-row">
        <div class="brand-mark">◈</div>
        <div>
            <div class="brand-title">Low-Light FAS</div>
            <div class="brand-subtitle">Face anti-spoofing & identity verification</div>
        </div>
    </div>

    <div class="system-badge">
        <span class="status-dot"></span>
        Anti-Spoofing + UniFace
    </div>
    """)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    # --------------------------------------------------------
    # Anti-Spoofing
    # --------------------------------------------------------

    original_directory = os.getcwd()

    os.chdir("antispoof_model")

    spoof_model = AntiSpoofPredict(0)
    cropper = CropImage()

    os.chdir(original_directory)

    # --------------------------------------------------------
    # UniFace
    # --------------------------------------------------------

    detector = RetinaFace(
        confidence_threshold=0.5
    )

    recognizer = ArcFace()

    return (
        spoof_model,
        cropper,
        detector,
        recognizer,
    )


# ============================================================
# ANTI-SPOOFING
# ============================================================

def get_spoof_score(
    image,
    model_test,
    cropper,
):

    model_dir = (
        "antispoof_model/resources/"
        "anti_spoof_models"
    )

    image_bbox = model_test.get_bbox(
        image
    )

    if sum(image_bbox) == 0:
        return None, 0.0

    prediction = np.zeros(
        (1, 3)
    )

    for model_name in os.listdir(
        model_dir
    ):

        if not model_name.endswith(
            ".pth"
        ):
            continue

        (
            h_input,
            w_input,
            model_type,
            scale,
        ) = parse_model_name(
            model_name
        )

        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }

        if scale is None:
            param["crop"] = False

        img = cropper.crop(
            **param
        )

        prediction += model_test.predict(
            img,
            os.path.join(
                model_dir,
                model_name,
            ),
        )

    label = np.argmax(
        prediction
    )

    score = (
        prediction[0][label] / 2.0
    )

    return label, score


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        '<div class="sidebar-brand">'
        'Identity Reference'
        '</div>')

    st.html(
        """
        <div class="sidebar-caption">
            Add a reference face, then compare a target
            frame using the project’s anti-spoofing and
            UniFace recognition stages.
        </div>
        """)

    st.html(
        '<div class="side-section">Master Reference</div>')

    ref_upload = st.file_uploader(
        "Upload reference image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        help=(
            "A frontal, high-resolution image "
            "is recommended."
        ),
    )

    ref_cv2 = None

    if ref_upload is not None:

        ref_bytes = np.asarray(
            bytearray(
                ref_upload.read()
            ),
            dtype=np.uint8,
        )

        ref_cv2 = cv2.imdecode(
            ref_bytes,
            cv2.IMREAD_COLOR,
        )

        if ref_cv2 is not None:

            st.success(
                "Reference image loaded"
            )

            st.image(
                cv2.cvtColor(
                    ref_cv2,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

    st.markdown("---")

    st.html(
        '<div class="side-section">Models in this project</div>')

    st.html(
        """
        <div class="model-group">
            <div class="model-group-title">Anti-Spoofing <span>2 local weights</span></div>
            <div class="model-group-note">antispoof_model/resources/anti_spoof_models</div>
            <div class="model-item">
                <strong>MiniFASNetV2</strong>
                2.7_80x80_MiniFASNetV2.pth
            </div>
            <div class="model-item">
                <strong>MiniFASNetV1SE</strong>
                4_0_0_80x80_MiniFASNetV1SE.pth
            </div>
        </div>

        <div class="model-group">
            <div class="model-group-title">UniFace <span>runtime models</span></div>
            <div class="model-item">
                <strong>Face Detection</strong>
                <span>Face location and landmarks</span>
            </div>
            <div class="model-item">
                <strong>Identity Matching</strong>
                <span>Face embeddings and similarity score</span>
            </div>
        </div>
        """)


# ============================================================
# INITIAL STATE
# ============================================================

if ref_cv2 is None:

    st.html(
        """
        <div class="panel" style="padding: 28px 22px; text-align: center;">
            <div style="font-size: 34px; margin-bottom: 10px;">◈</div>
            <div style="font-size: 22px; font-weight: 800; color: #edf7ff;">Awaiting identity enrollment</div>
            <div style="color: #7d8ea0; font-size: 13px; margin-top: 10px;">
                Upload a master reference image from the sidebar to begin verification.
            </div>
        </div>
        """)

    st.stop()


# ============================================================
# TARGET INPUT
# ============================================================

st.html(
    """
    <div class="section-row">

        <div class="section-title">
            Verification Workspace
        </div>

        <div class="section-caption">
            Master → Liveness → Recognition
        </div>

    </div>
    """)

test_upload = st.file_uploader(
    "Target frame",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    help=(
        "The submitted frame will be evaluated "
        "for liveness and identity."
    ),
)


if test_upload is None:

    st.html(
        """
        <div class="panel" style="padding: 28px 22px; text-align: center;">
            <div style="font-size: 15px; font-weight: 700; color: #edf7ff;">Ready for verification</div>
            <div style="font-size: 12px; color: #7c8f9d; margin-top: 8px;">
                Upload a target frame to begin the biometric security pipeline.
            </div>
        </div>
        """)

    st.stop()


# ============================================================
# READ TARGET IMAGE
# ============================================================

test_bytes = np.asarray(
    bytearray(
        test_upload.read()
    ),
    dtype=np.uint8,
)

test_cv2 = cv2.imdecode(
    test_bytes,
    cv2.IMREAD_COLOR,
)

if test_cv2 is None:

    st.error(
        "Unable to decode the uploaded target image."
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner(
    "Loading biometric models..."
):

    (
        spoof_model,
        cropper,
        detector,
        recognizer,
    ) = load_models()


# ============================================================
# PIPELINE HEADER
# ============================================================

st.html(
    """
    <div class="pipeline">

        <div class="pipeline-node">
            01 · Anti-Spoofing
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-node">
            02 · UniFace Detection
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-node">
            03 · Identity Match
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-node">
            04 · Identity Decision
        </div>

    </div>
    """)


# ============================================================
# MAIN COMPARISON AREA
# ============================================================

col_master, col_target, col_verdict = st.columns(
    [1, 1, 1.15],
    gap="large",
)


# ============================================================
# MASTER IMAGE
# ============================================================

with col_master:

    with st.container(border=True, key="trusted_identity_view"):

        st.html('<div class="image-label">Trusted Identity</div>')

        st.image(
            cv2.cvtColor(
                ref_cv2,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

        st.caption("REGISTERED REFERENCE")


# ============================================================
# TARGET IMAGE
# ============================================================

with col_target:

    with st.container(border=True, key="submitted_frame_view"):

        st.html('<div class="image-label">Submitted Frame</div>')

        st.image(
            cv2.cvtColor(
                test_cv2,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

        st.caption("UNTRUSTED INPUT")


# ============================================================
# VERDICT
# ============================================================

with col_verdict:

    st.html(
        '<div class="image-label">Security Decision</div>')

    # --------------------------------------------------------
    # Default telemetry state
    # --------------------------------------------------------

    spoof_label = None
    spoof_score = 0.0

    similarity = None
    similarity_threshold = 0.40

    ref_emb = None
    test_emb = None

    ref_faces = []
    test_faces = []

    # --------------------------------------------------------
    # Pipeline execution
    # --------------------------------------------------------

    with st.status(
        "Executing verification pipeline...",
        expanded=True,
    ) as pipeline_status:

        st.write(
            "01  •  Running local anti-spoofing models"
        )

        spoof_label, spoof_score = get_spoof_score(
            test_cv2,
            spoof_model,
            cropper,
        )

        # ====================================================
        # STEP 2
        # ====================================================

        st.write(
            "02  •  Detecting facial geometry with UniFace"
        )

        ref_faces = detector.detect(
            ref_cv2
        )

        test_faces = detector.detect(
            test_cv2
        )

        # ====================================================
        # STEP 3
        # ====================================================

        if (
            spoof_label == 1
            and ref_faces
            and test_faces
        ):

            st.write(
                "03  •  Generating identity embeddings"
            )

            ref_face = ref_faces[0]
            test_face = test_faces[0]

            try:

                ref_emb = (
                    recognizer.get_normalized_embedding(
                        ref_cv2,
                        ref_face.landmarks,
                    )
                )

                test_emb = (
                    recognizer.get_normalized_embedding(
                        test_cv2,
                        test_face.landmarks,
                    )
                )

            except Exception as error:

                pipeline_status.update(
                    label="Pipeline error",
                    state="error",
                    expanded=True,
                )

                st.error(
                    "Identity embedding extraction failed."
                )

                st.exception(error)

            # ================================================
            # STEP 4
            # ================================================

            if (
                ref_emb is not None
                and test_emb is not None
            ):

                st.write(
                    "04  •  Computing cosine similarity"
                )

                similarity = compute_similarity(
                    ref_emb,
                    test_emb,
                )

        pipeline_status.update(
            label="Verification complete",
            state="complete",
            expanded=False,
        )

    # ========================================================
    # RESULT: SPOOF
    # ========================================================

    if spoof_label is None:

        st.html(
            """
            <div class="decision-panel">
                <div class="decision-kicker">Liveness</div>
                <div class="decision-title decision-danger">Face Not Detected</div>
                <div class="decision-copy">
                    The anti-spoofing subsystem could not obtain a usable facial region from the submitted frame.
                </div>
            </div>
            """)

    # ========================================================
    # RESULT: SPOOF DETECTED
    # ========================================================

    elif spoof_label != 1:

        st.html(
            f"""
            <div class="decision-panel">

                <div class="decision-kicker">
                    Security Decision
                </div>

                <div class="decision-title decision-danger">
                    Presentation Attack
                </div>

                <div class="decision-copy">
                    The liveness classifier rejected the
                    submitted frame as a spoof attempt.
                </div>

                <div class="spoof-box">

                    <div class="score-label">
                        Liveness Confidence
                    </div>

                    <div class="score-value">
                        {spoof_score:.2%}
                    </div>

                </div>

            </div>
            """)

    # ========================================================
    # RESULT: FACE DETECTION FAILURE
    # ========================================================

    elif (
        not ref_faces
        or not test_faces
    ):

        st.html(
            """
            <div class="decision-panel">

                <div class="decision-kicker">
                    Recognition
                </div>

                <div class="decision-title decision-danger">
                    Face Alignment Failed
                </div>

                <div class="decision-copy">
                    UniFace could not locate a valid
                    facial landmark configuration in the
                    reference or target image.
                </div>

            </div>
            """)

    # ========================================================
    # RESULT: EMBEDDING FAILURE
    # ========================================================

    elif similarity is None:

        st.html(
            """
            <div class="decision-panel">

                <div class="decision-kicker">
                    Recognition
                </div>

                <div class="decision-title decision-danger">
                    Embedding Extraction Failed
                </div>

                <div class="decision-copy">
                    Facial geometry was detected, but
                    the identity matching stage did not
                    produce valid vectors.
                </div>

            </div>
            """)

    # ========================================================
    # RESULT: SUCCESS / MATCH
    # ========================================================

    else:

        is_match = (
            similarity
            >= similarity_threshold
        )

        # ----------------------------------------------------
        # LIVE STATUS
        # ----------------------------------------------------

        st.html(
            f"""
            <div class="decision-panel">

                <div class="decision-kicker">
                    Security Decision
                </div>

                <div class="decision-title decision-live">
                    Live Subject Detected
                </div>

                <div class="decision-copy">
                    The submitted frame passed the
                    anti-spoofing stage and produced
                    valid biometric embeddings.
                </div>

            </div>
            """)

        # ----------------------------------------------------
        # SCORE CARDS
        # ----------------------------------------------------

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            st.html(
                f"""
                <div class="score-card">

                    <div class="score-label">
                        Liveness
                    </div>

                    <div class="score-value">
                        {spoof_score:.2%}
                    </div>

                    <div class="score-detail">
                        Local anti-spoofing ensemble
                    </div>

                </div>
                """)

        with score_col2:

            st.html(
                f"""
                <div class="score-card">

                    <div class="score-label">
                        Similarity
                    </div>

                    <div class="score-value">
                        {similarity:.4f}
                    </div>

                    <div class="score-detail">
                        Match threshold: {similarity_threshold:.2f}
                    </div>

                </div>
                """)

        # ----------------------------------------------------
        # FINAL IDENTITY RESULT
        # ----------------------------------------------------

        if is_match:

            st.html(
                """
                <div class="verified-box">

                    <div style="
                        font-size: 20px;
                        font-weight: 800;
                        color: #69e2ae;
                    ">
                        IDENTITY VERIFIED
                    </div>

                    <div style="
                        margin-top: 6px;
                        color: #82908c;
                        font-size: 12px;
                    ">
                        Liveness passed and face similarity
                        exceeded the configured threshold.
                    </div>

                </div>
                """)

        else:

            st.html(
                """
                <div class="mismatch-box">

                    <div style="
                        font-size: 20px;
                        font-weight: 800;
                        color: #ff8585;
                    ">
                        IDENTITY MISMATCH
                    </div>

                    <div style="
                        margin-top: 6px;
                        color: #917e81;
                        font-size: 12px;
                    ">
                        The subject appears live, but the
                        biometric embedding does not satisfy
                        the configured similarity threshold.
                    </div>

                </div>
                """)


# ============================================================
# TECHNICAL TELEMETRY
# ============================================================

st.html(
    """
    <div class="section-row" style="margin-top: 30px;">

        <div class="section-title">
            Technical Telemetry
        </div>

        <div class="section-caption">
            Model-level diagnostic information
        </div>

    </div>
    """)


with st.expander(
    "View pipeline metrics and configuration",
    expanded=False,
):

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    # --------------------------------------------------------
    # LIVENESS
    # --------------------------------------------------------

    with metric1:

        st.metric(
            label="Liveness Score",
            value=(
                f"{spoof_score:.2%}"
                if spoof_label is not None
                else "N/A"
            ),
        )

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    with metric2:

        st.metric(
            label="Reference Face",
            value=(
                "Detected"
                if ref_faces
                else "Not detected"
            ),
        )

    # --------------------------------------------------------
    # TARGET DETECTION
    # --------------------------------------------------------

    with metric3:

        st.metric(
            label="Target Face",
            value=(
                "Detected"
                if test_faces
                else "Not detected"
            ),
        )

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    with metric4:

        st.metric(
            label="Face Similarity",
            value=(
                f"{similarity:.4f}"
                if similarity is not None
                else "N/A"
            ),
        )

    st.markdown("---")

    telemetry_left, telemetry_right = st.columns(2)

    with telemetry_left:

        st.markdown(
            """
            **Detection**

            - Stage: `UniFace detection`
            - Detection threshold: `0.50`
            - Reference faces: `{}`
            - Target faces: `{}`
            """.format(
                len(ref_faces),
                len(test_faces),
            )
        )

    with telemetry_right:

        st.markdown(
            """
            **Recognition**

            - Stage: `UniFace identity matching`
            - Embeddings: `{}`
            - Similarity metric: `Cosine`
            - Match threshold: `{}`
            """.format(
                (
                    "Generated"
                    if ref_emb is not None
                    and test_emb is not None
                    else "Unavailable"
                ),
                similarity_threshold,
            )
        )

    st.markdown("---")

    st.markdown(
        """
        **Pipeline**

        `Target Frame`
        →
        `Anti-Spoofing`
        →
        `UniFace Detection`
        →
        `Identity Match`
        →
        `Cosine Similarity`
        →
        `Identity Decision`
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        margin-top: 36px;
        padding-top: 16px;
        border-top: 1px solid #1b252e;
        color: #56636f;
        font-size: 11px;
        text-align: center;
    ">
        LOW-LIGHT FAS • MINIFASNET ANTI-SPOOFING • UNIFACE DETECTION + IDENTITY MATCHING
    </div>
    """)
