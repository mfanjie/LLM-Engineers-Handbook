from loguru import logger
from zenml import step

from llm_engineering.domain.base import VectorDBDataModel


@step
def load_to_vector_db(
    documents: list,
) -> None:
    logger.info(f"Loading # documents: {len(documents)}")

    grouped_documents = VectorDBDataModel.group_by_collection(documents)
    for data_model_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {data_model_class.get_collection_name()}")
        data_model_class.bulk_insert(documents)