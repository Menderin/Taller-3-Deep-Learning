"""Streamlit user interface for the trained multitask model."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

from PIL import Image
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.inference.face_detector import FaceDetector
from src.inference.predictor import CNNPredictor
from src.utils import resolve_device
from Frontend.ui_styles import apply_custom_css, get_base64_of_bin_file

FRONTEND_DIR = Path(__file__).resolve().parent
RESULTS_DIR = FRONTEND_DIR / "assets" / "results"


@st.cache_resource(show_spinner="Cargando modelo...")
def load_predictor(checkpoint_path: str, device_name: str) -> CNNPredictor:
    """Load model weights once per Streamlit process."""

    return CNNPredictor.from_checkpoint(
        checkpoint_path,
        resolve_device(device_name),
    )


def run_app() -> None:
    """Render the upload/camera, face detection and prediction workflow."""
    st.set_page_config(
        page_title="UTKFace: Genero y Edad",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    apply_custom_css()

    config = AppConfig.from_env()
    device = resolve_device(config.device)

    # Custom Header Navigation (Replacing Sidebar)
    st.markdown("<h1 style='text-align: center; color: white;'>FaceScope: Estimación Multitarea de Género y Edad</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-subtitle'>Taller 3 Deep Learning</p>",
        unsafe_allow_html=True,
    )
    
    experiment_name = config.cnn_checkpoint.parent.name
    model_labels = {
        "resnet_finetuning_lambda_high": (
            "ResNet18 fine-tuning (&lambda;<sub>edad</sub> = 0,1)"
        ),
        "resnet_finetuning_unfreeze_more": "ResNet18 fine-tuning ampliado",
        "cnn_base": "CNN multitarea",
        "models": "ResNet18 fine-tuning (&lambda;<sub>edad</sub> = 0,1)",
    }
    model_label = model_labels.get(
        experiment_name,
        experiment_name.replace("_", " ").title(),
    )
    st.markdown(
        f"<div class='active-model'>Modelo activo: {model_label} &middot; "
        f"Dispositivo: {device}</div>",
        unsafe_allow_html=True,
    )

    tab_prediccion, tab_analytics = st.tabs(["Predicción", "Desempeño de Modelos"])

    with tab_prediccion:
        render_prediction_page(config, device)
        
    with tab_analytics:
        render_analytics_page()

    render_footer()


def render_footer() -> None:
    logo_path = FRONTEND_DIR / "assets" / "github-logo.png"
    logo_html = ""
    if logo_path.exists():
        logo_base64 = get_base64_of_bin_file(logo_path)
        logo_html = (
            f'<img class="github-logo" '
            f'src="data:image/png;base64,{logo_base64}" '
            f'alt="GitHub">'
        )

    st.markdown(
        f"""
        <footer class="app-footer">
            <span>Desarrollado por Victor Jopia</span>
            <span class="footer-separator">&middot;</span>
            <a
                href="https://github.com/Menderin/Taller-3-Deep-Learning"
                target="_blank"
                rel="noopener noreferrer"
            >
                {logo_html}
                Repositorio en GitHub
            </a>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_page(config: AppConfig, device) -> None:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Entrada")
        uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
        captured_file = st.camera_input("O captura una imagen con la camara")
        source_file = captured_file if captured_file is not None else uploaded_file

        if source_file is None:
            st.info("Sube o captura una imagen para comenzar.")
            return

        image = Image.open(source_file).convert("RGB")
        detector = FaceDetector()
        detection = detector.detect_largest(image)

        if detection is None:
            st.image(image, use_container_width=True)
            st.warning("No se detecto un rostro frontal. Prueba otra imagen.")
            return
        
        st.image(
            detector.draw_box(image, detection),
            caption="Rostro detectado",
            use_container_width=True,
        )

    with col2:
        st.subheader("Resultado")
        st.image(detection.crop, width=200, caption="Rostro recortado")
        
        if st.button("Ejecutar modelo", type="primary", use_container_width=True):
            try:
                predictor = load_predictor(
                    str(config.cnn_checkpoint),
                    str(device),
                )
                prediction = predictor.predict(detection.crop)
                
                # Metrics layout
                st.markdown("### Prediccion")
                m1, m2 = st.columns(2)
                m1.metric("Genero predicho", prediction.gender_label)
                confidence_percent = prediction.gender_confidence * 100
                confidence_label = (
                    ">99.9%"
                    if confidence_percent >= 99.95
                    else f"{confidence_percent:.1f}%"
                )
                m2.metric("Confianza estimada", confidence_label)
                
                m3, _ = st.columns(2)
                m3.metric("Edad estimada", f"{prediction.estimated_age:.1f} años")
                
                st.caption(
                    "Estas salidas reflejan las etiquetas binarias y sesgos del dataset "
                    "UTKFace; la confianza no representa certeza y no debe interpretarse "
                    "como identidad de genero."
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                st.error(str(error))

def render_analytics_page() -> None:
    st.subheader("Analisis de los modelos entrenados y sus metricas de rendimiento")

    report_path = RESULTS_DIR / "all_experiments_comparison.csv"
    if not report_path.exists():
        st.error(f"No se encontro el archivo de reporte en: {report_path}")
        return

    # Read and parse CSV
    with open(report_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Format data for display
    formatted_data = []
    for row in data:
        try:
            formatted_data.append({
                "Experimento": row.get("experiment_name", ""),
                "Estrategia": row.get("strategy_name", ""),
                "Accuracy": float(row.get("gender_accuracy", 0)),
                "MAE": float(row.get("age_mae", 100)),
                "R2": float(row.get("age_r2", 0))
            })
        except ValueError:
            continue

    st.markdown("#### Top 5 Modelos (Por Accuracy de Genero)")
    top_gender = sorted(formatted_data, key=lambda x: x["Accuracy"], reverse=True)[:5]
    st.dataframe(top_gender, use_container_width=True)

    st.markdown("#### Top 5 Modelos (Por Menor MAE de Edad)")
    top_age = sorted(formatted_data, key=lambda x: x["MAE"], reverse=False)[:5]
    st.dataframe(top_age, use_container_width=True)

    # Display pre-generated plots
    plots_dir = RESULTS_DIR
    
    st.markdown("#### Visualizaciones Globales")
    c1, c2 = st.columns(2)
    
    plot_gender = plots_dir / "global_gender_comparison.png"
    if plot_gender.exists():
        c1.image(Image.open(plot_gender), caption="Comparacion de Genero", use_container_width=True)
        
    plot_age = plots_dir / "global_age_comparison.png"
    if plot_age.exists():
        c2.image(Image.open(plot_age), caption="Comparacion de Edad", use_container_width=True)

    plot_tradeoff = plots_dir / "global_multitask_tradeoff.png"
    if plot_tradeoff.exists():
        st.image(Image.open(plot_tradeoff), caption="Tradeoff Multitarea", use_container_width=True)


if __name__ == "__main__":
    run_app()
