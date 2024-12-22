from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger
import os
import subprocess
import sys

try:
    from sagemaker.huggingface import HuggingFace
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

    # Verify Hugging Face credentials
    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # Set environment variables
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HUGGINGFACE_ACCESS_TOKEN
    if hasattr(settings, 'COMET_API_KEY'):
        os.environ["COMET_API_KEY"] = settings.COMET_API_KEY
    if hasattr(settings, 'COMET_PROJECT'):
        os.environ["COMET_PROJECT_NAME"] = settings.COMET_PROJECT

    # Prepare arguments for the training script
    args = [
        "--finetuning_type", finetuning_type,
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--learning_rate", str(learning_rate),
        "--dataset_huggingface_workspace", dataset_huggingface_workspace,
        "--model_output_huggingface_workspace", huggingface_user,
    ]
    
    if is_dummy:
        args.extend(["--is_dummy", "True"])

    # Run the training script
    script_path = finetuning_dir / "finetune.py"
    try:
        # Option 1: Run as a module
        subprocess.run([sys.executable, str(script_path)] + args, check=True)
        
        # Option 2: Import and run directly (alternative approach)
        '''
        import importlib.util
        spec = importlib.util.spec_from_file_location("finetune", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(args)  # Assuming your finetune.py has a main() function
        '''
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    run_finetuning_on_sagemaker()

