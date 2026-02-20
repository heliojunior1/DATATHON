"""
Definição das entidades do Feature Store.

Entidade principal: Aluno (identificado pelo RA).
"""
from feast import Entity, ValueType

# Entidade Aluno — chave primária do Feature Store
aluno = Entity(
    name="aluno",
    join_keys=["aluno_id"],
    description="Aluno da Associação Passos Mágicos, identificado pelo RA.",
)
