"""
Utilitário de tratamento de erros para routers FastAPI.

Centraliza o padrão repetido de try/except nos endpoints,
eliminando duplicação e padronizando os códigos de status HTTP.
"""
import functools
import logging

from fastapi import HTTPException


def handle_route_errors(logger: logging.Logger):
    """
    Decorator que envolve endpoints FastAPI com tratamento de erros padronizado.

    Mapeia exceções comuns para os respectivos códigos HTTP:
    - FileNotFoundError → 503 (modelo ou recurso não encontrado)
    - ImportError       → 501 (dependência opcional não instalada)
    - ValueError        → 400 (parâmetros inválidos)
    - Exception         → 500 (erro interno genérico)

    Args:
        logger: Logger do módulo chamador (use setup_logger(__name__)).

    Exemplo de uso:
        @router.get("/endpoint")
        @handle_route_errors(logger)
        async def my_endpoint():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Recurso não encontrado: {e}",
                )
            except ImportError as e:
                raise HTTPException(
                    status_code=501,
                    detail=f"Dependência não instalada: {e}",
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Erro em {func.__name__}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return wrapper
    return decorator
