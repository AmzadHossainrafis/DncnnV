import numpy as np
import torch
import sys
import mlflow
import mlflow.pytorch
from dncnn.components.dataloader import DataLoader, config
from dncnn.components.model import DnCNN
from tqdm import tqdm
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)
from dncnn.utils.logger import logger
from dncnn.utils.exception import CustomException, InvalidFormatError
from dncnn.utils.common import count_items_in_directory

# Set the config
try:
    eval_config = config["Test_DL_config"]
except KeyError as e:
    logger.error(f"Configuration key error: {e}")
    raise CustomException(f"Configuration key error: {e}", sys)

# Create data loader
try:
    train_dataloader = DataLoader(
        eval_config["test_hr_dir"],
        batch_size=eval_config["batch_size"],
        shuffle=eval_config["shuffle"],
        num_workers=eval_config["num_workers"],
        transform=eval_config["transform"],
        random_blur=eval_config["random_blur"],
    )
except KeyError as e:
    logger.error(f"Configuration key error while creating DataLoader: {e}")
    raise CustomException(f"Configuration key error: {e}", sys)
except Exception as e:
    logger.error(f"Error while creating DataLoader: {e}")
    raise CustomException(f"Error while creating DataLoader: {e}", sys)

# Select model
eval_model = DnCNN()

# Load model weights
try:
    eval_weights = config["evaluation_tracker"]["model_path"]
    eval_model.load_state_dict(torch.load(eval_weights, weights_only=True))
except KeyError as e:
    logger.error(f"Configuration key error while loading model weights: {e}")
    raise CustomException(f"Configuration key error: {e}", sys)
except Exception as e:
    logger.error(f"Error while loading model weights: {e}")
    raise CustomException(f"Error while loading model weights: {e}", sys)

eval_model.eval()


def evaluate(
    model=eval_model,
    eval_dataloader=train_dataloader,
    device=config["evaluation_tracker"]["device"],
):
    """
    Evaluates the model on the provided dataloader and logs the metrics using MLflow.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        eval_dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        device (str): The device to run the evaluation on (e.g., "cuda" or "cpu").

    Returns:
        tuple: A tuple containing the final mean squared error (MSE) loss, SSIM score, and PSNR score.
    """
    model = model.to(device)
    criterion = torch.nn.MSELoss()  # Instantiate MSELoss
    ssim_metric = SSIM(data_range=1.0).to(device)  # Instantiate SSIM metric
    psnr_metric = PSNR(data_range=1.0).to(device)  # Instantiate PSNR metric

    eval_loss_per_epoch = []
    ssim_scores = []
    psnr_scores = []

    try:
        num_items = count_items_in_directory(eval_config["test_hr_dir"])
        num_batches = num_items // eval_config["batch_size"]
    except Exception as e:
        logger.error(f"Error counting items in directory: {e}")
        raise CustomException(f"Error counting items in directory: {e}", sys)

    eval_bar = tqdm(enumerate(eval_dataloader), total=num_batches, desc="Evaluating")

    try:
        with torch.inference_mode():
            with mlflow.start_run() as run:
                for idx, (lr, hr) in eval_bar:
                    hr = hr.to(device)
                    lr = lr.to(device)
                    sr = model(lr)

                    # Calculate loss
                    loss = criterion(sr, hr)
                    eval_loss_per_epoch.append(loss.item())

                    # Calculate SSIM and PSNR
                    ssim_score = ssim_metric(sr, hr)
                    psnr_score = psnr_metric(sr, hr)

                    ssim_scores.append(ssim_score.item())
                    psnr_scores.append(psnr_score.item())

                    # Update tqdm description with current metrics
                    eval_bar.set_description(
                        f"Iter {idx + 1} - Loss: {loss.item():.4f}, SSIM: {ssim_score.item():.4f}, PSNR: {psnr_score.item():.4f}"
                    )

                    # Log metrics for the current iteration
                    mlflow.log_metrics(
                        {
                            "Iteration_Loss": loss.item(),
                            "Iteration_SSIM": ssim_score.item(),
                            "Iteration_PSNR": psnr_score.item(),
                        },
                        step=idx + 1,
                    )

                # Calculate and log final metrics
                final_loss = np.mean(eval_loss_per_epoch)
                final_ssim = np.mean(ssim_scores)
                final_psnr = np.mean(psnr_scores)

                logger.info(
                    f"\nFinal Eval Metrics: MSE Loss={final_loss:.4f}, SSIM={final_ssim:.4f}, PSNR={final_psnr:.4f}"
                )

                mlflow.log_metrics(
                    {
                        "Final_Eval_Loss": final_loss,
                        "Final_Eval_SSIM": final_ssim,
                        "Final_Eval_PSNR": final_psnr,
                    }
                )

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise CustomException(f"Error during evaluation: {e}", sys)

    return final_loss, final_ssim, final_psnr


if __name__ == "__main__":

    logger.info("--------------- STARTNG EVALUATION --------------- ")
    try:
        # Run evaluation
        evaluate(
            model=eval_model,
            eval_dataloader=train_dataloader,
        )
    except CustomException as e:
        logger.error(f"Custom exception occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
