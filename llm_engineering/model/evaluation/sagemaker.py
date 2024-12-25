from pathlib import Path
from huggingface_hub import HfApi
from loguru import logger
import sys
import os  # 添加用于设置环境变量
from llm_engineering import settings

evaluation_dir = Path(__file__).resolve().parent
evaluation_requirements_path = evaluation_dir / "requirements.txt"


def run_evaluation_on_sagemaker(is_dummy: bool = True) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.OPENAI_API_KEY, "OpenAI API key is required."
    # 移除 AWS role 检查，因为本地执行不需要
    # assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    if not evaluation_dir.exists():
        raise FileNotFoundError(f"The directory {evaluation_dir} does not exist.")
    if not evaluation_requirements_path.exists():
        raise FileNotFoundError(f"The file {evaluation_requirements_path} does not exist.")

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # 保持原有的环境变量设置结构
    env = {
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "DATASET_HUGGINGFACE_WORKSPACE": huggingface_user,
        "MODEL_HUGGINGFACE_WORKSPACE": huggingface_user,
    }
    if is_dummy:
        env["IS_DUMMY"] = "True"

    # 设置环境变量
    for key, value in env.items():
        os.environ[key] = value

    # 直接执行 evaluate.py
    evaluate_script = evaluation_dir / "evaluate.py"
    exec(open(evaluate_script).read())


if __name__ == "__main__":
    run_evaluation_on_sagemaker()

