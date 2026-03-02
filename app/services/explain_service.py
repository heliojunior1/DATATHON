import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

from app.utils.helpers import setup_logger
from app.services.training.model_registry import create_model

logger = setup_logger(__name__)


def fig_to_base64(fig: plt.Figure) -> str:
    """Converte uma figura do Matplotlib para string base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class ExplainService:
    """Serviço para gerar explicabilidade visual (XAI) de modelos treinados."""

    @staticmethod
    def explain_tree_model(model, X: pd.DataFrame, max_display: int = 10) -> dict:
        """
        Gera um SHAP Summary Plot para modelos de árvore (XGBoost, LightGBM, CatBoost).
        """
        try:
            # SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Para classificação binária, shap_values pode ser uma lista. Pegamos a classe positiva [1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Inicializa a figura
            fig = plt.figure(figsize=(10, max_display * 0.5 + 2))
            
            # Plot
            shap.summary_plot(
                shap_values, 
                X, 
                max_display=max_display, 
                show=False,
                plot_size=None # permite ao plt gerir o tamanho
            )
            
            plt.title("SHAP Summary: Impacto das Features", pad=20, fontsize=14)
            plt.tight_layout()
            
            img_b64 = fig_to_base64(fig)
            
            return {
                "type": "shap_summary",
                "image_base64": img_b64,
                "description": "Gráfico SHAP. Cores representam o valor da feature (alto/baixo), posição horizontal representa impacto na predição."
            }

        except Exception as e:
            logger.error(f"Erro ao gerar SHAP plot: {e}", exc_info=True)
            raise RuntimeError(f"Falha na explicabilidade Tree/SHAP: {str(e)}")

    @staticmethod
    def explain_svm_boundary(model, X: pd.DataFrame, y: np.ndarray | pd.Series) -> dict:
        """
        Gera o plot da Fronteira de Decisão do SVM reduzindo os dados para 2D com PCA.
        NOTA: Treina um modelo SVM 'dummy' apenas para visualização 2D.
        """
        try:
            # 1. Redução de Dimensionalidade (PCA) para 2 features
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Precisamos extrair os parâmetros do modelo original para criar um similar
            # O model é um Pipeline (StandardScaler + SVC)
            if hasattr(model, "named_steps") and "svc" in model.named_steps:
                svc_original = model.named_steps["svc"]
                C = svc_original.C
                kernel = svc_original.kernel
                gamma = svc_original.gamma
            else:
                C, kernel, gamma = 1.0, 'rbf', 'scale'

            # Treina modelo de visualização 2D
            from sklearn.svm import SVC
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            dummy_model = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(C=C, kernel=kernel, gamma=gamma, random_state=42))
            ])
            
            dummy_model.fit(X_pca, np.array(y))
            
            # Cria a figura
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Mlxtend decide regions setup
            plot_decision_regions(
                X=X_pca, 
                y=np.array(y).astype(int), 
                clf=dummy_model, 
                legend=2, 
                ax=ax,
                colors="#1f77b4,#ff7f0e",
                scatter_kwargs={'alpha': 0.6, 's': 40, 'edgecolor': "white"}
            )
            
            # Anotações do eixo representam os Componentes Principais, não as features orignais
            explained_variance = pca.explained_variance_ratio_ * 100
            ax.set_xlabel(f"Componente Principal 1 ({explained_variance[0]:.1f}%)")
            ax.set_ylabel(f"Componente Principal 2 ({explained_variance[1]:.1f}%)")
            ax.set_title("Fronteira de Decisão (Visão 2D Simplificada - PCA)", fontsize=14)
            plt.tight_layout()
            
            img_b64 = fig_to_base64(fig)
            
            return {
                "type": "decision_boundary",
                "image_base64": img_b64,
                "description": "Fronteira de Decisão 2D. Os dados foram reduzidos usando PCA para visualização da separação das classes."
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar SVM Boundary plot: {e}", exc_info=True)
            raise RuntimeError(f"Falha na explicabilidade SVM/Boundary: {str(e)}")

    @staticmethod
    def explain_linear_model(model, feature_names: list[str]) -> dict:
        """
        Retorna o gráfico de barras dos coeficientes do modelo linear (Regressão Logística).
        """
        try:
            # Pega os coeficientes (LogisticRegression guarda coef_ em formato (n_classes, n_features))
            if not hasattr(model, 'coef_'):
                raise ValueError("Modelo não possui o atributo coef_ (não é linear puro).")
                
            coefs = model.coef_[0] # coeficientes da classe 1
            
            # Cria dataframe para facilitar ordenação usando seaborn
            df_coef = pd.DataFrame({
                "Feature": feature_names,
                "Coeficiente": coefs,
                "Importância Absoluta": np.abs(coefs)
            })
            
            # Ordenar pelos mais importantes
            df_coef = df_coef.sort_values(by="Importância Absoluta", ascending=False).head(20)
            
            fig, ax = plt.subplots(figsize=(10, max(6, len(df_coef) * 0.4)))
            
            sns.barplot(
                data=df_coef, 
                x="Coeficiente", 
                y="Feature", 
                hue=df_coef["Coeficiente"] > 0, 
                palette={True: "#2ca02c", False: "#d62728"}, # Verde p/ positivo, Vermelho p/ negativo
                legend=False,
                ax=ax
            )
            
            ax.set_title("Top 20 Features - Peso no Hiperplano (Coeficientes)", fontsize=14)
            ax.set_xlabel("Valor do Coeficiente Matemático")
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            plt.tight_layout()
            
            img_b64 = fig_to_base64(fig)
            
            return {
                "type": "linear_coefficients",
                "image_base64": img_b64,
                "description": "Impacto matemático das features no modelo linear. Barras verdes empurram em direção à classe 1, barras vermelhas em direção à classe 0."
            }

        except Exception as e:
            logger.error(f"Erro ao gerar Linear Coef plot: {e}", exc_info=True)
            raise RuntimeError(f"Falha na explicabilidade Linear/Coef: {str(e)}")
