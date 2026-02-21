"""
Schemas Pydantic para predição de risco de defasagem.
"""
from pydantic import BaseModel, Field
from typing import Optional


class StudentInput(BaseModel):
    """Dados de entrada de um aluno para previsão de risco de defasagem."""

    # Indicadores PEDE (obrigatórios)
    IAA: float = Field(..., ge=0, le=10, description="Indicador de Auto Avaliação (0-10)")
    IEG: float = Field(..., ge=0, le=10, description="Indicador de Engajamento (0-10)")
    IPS: float = Field(..., ge=0, le=10, description="Indicador Psicossocial (0-10)")
    IDA: float = Field(..., ge=0, le=10, description="Indicador de Aprendizagem (0-10)")
    IPV: float = Field(..., ge=0, le=10, description="Indicador de Ponto de Virada (0-10)")
    IAN: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Adequação ao Nível (0-10) — desativado por padrão (data leakage)")
    INDE_22: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional", alias="INDE 22")

    # Notas (Inglês removido — 67% nulos)
    Matem: float = Field(..., ge=0, le=10, description="Nota de Matemática")
    Portug: float = Field(..., ge=0, le=10, description="Nota de Português")
    Tem_nota_ingles: Optional[int] = Field(None, ge=0, le=1, description="1 se tem nota de inglês, 0 se não")

    # Demográficas
    Idade_22: int = Field(..., ge=5, le=25, description="Idade do aluno em 2022", alias="Idade 22")
    Genero: str = Field(..., description="Gênero: 'Menina' ou 'Menino'", alias="Gênero")
    Instituicao_ensino: str = Field(
        ...,
        description="Instituição: 'Escola Pública', 'Rede Decisão' ou 'Escola JP II'",
        alias="Instituição de ensino",
    )
    Ano_ingresso: int = Field(..., ge=2010, le=2023, description="Ano de ingresso na Passos Mágicos", alias="Ano ingresso")

    # Fase (nova)
    Fase: Optional[str] = Field(None, description="Fase do aluno: 'Alfa', 'Fase 1'...'Fase 8'")

    # Classificação Pedras (opcionais — podem ter nulos)
    Pedra_20: Optional[str] = Field(None, description="Classificação Pedra 2020", alias="Pedra 20")
    Pedra_21: Optional[str] = Field(None, description="Classificação Pedra 2021", alias="Pedra 21")
    Pedra_22: str = Field(..., description="Classificação Pedra 2022", alias="Pedra 22")

    Atingiu_PV: str = Field(..., description="Atingiu Ponto de Virada: 'Sim' ou 'Não'", alias="Atingiu PV")
    Indicado: str = Field(..., description="Indicado para Bolsa: 'Sim' ou 'Não'")

    # Rankings (opcionais para alunos novos)
    Cf: Optional[int] = Field(None, ge=0, description="Classificação na Fase (opcional para alunos novos)")
    Ct: Optional[int] = Field(None, ge=0, description="Classificação na Turma (opcional para alunos novos)")
    N_Av: int = Field(..., ge=1, le=4, description="Número de Avaliações", alias="Nº Av")

    # Destaques
    Destaque_IEG: Optional[str] = Field("Não", description="Destaque IEG", alias="Destaque IEG")
    Destaque_IDA: Optional[str] = Field("Não", description="Destaque IDA", alias="Destaque IDA")
    Destaque_IPV: Optional[str] = Field("Não", description="Destaque IPV", alias="Destaque IPV")

    # Recomendações dos avaliadores
    Rec_av1_encoded: Optional[int] = Field(0, ge=0, le=4, description="Rec. Avaliador 1 codificada")
    Rec_av2_encoded: Optional[int] = Field(0, ge=0, le=4, description="Rec. Avaliador 2 codificada")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Resposta de previsão para um aluno."""
    prediction: int = Field(..., description="0 = Sem Risco, 1 = Em Risco")
    probability: float = Field(..., description="Probabilidade de risco (0 a 1)")
    risk_level: str = Field(..., description="Nível de risco: Muito Baixo, Baixo, Moderado, Alto, Muito Alto")
    label: str = Field(..., description="Label descritivo: 'Em Risco de Defasagem' ou 'Sem Risco'")
    top_factors: list[dict] = Field(default_factory=list, description="Top 5 features mais importantes")
    model_id: Optional[str] = Field(None, description="ID do modelo usado na predição")
    latency_ms: Optional[float] = Field(None, description="Latência de inferência em milissegundos")
    prediction_id: Optional[str] = Field(None, description="UUID da predição para feedback de concept drift")


class BatchPredictionRequest(BaseModel):
    """Request para previsão em lote."""
    students: list[StudentInput] = Field(..., min_length=1, max_length=100, description="Lista de alunos")
    model_id: Optional[str] = Field(None, description="ID do modelo a usar (None = mais recente)")


class BatchPredictionResponse(BaseModel):
    """Resposta de previsão em lote."""
    predictions: list[PredictionResponse]
    total: int
    risk_count: int
    risk_rate: float
