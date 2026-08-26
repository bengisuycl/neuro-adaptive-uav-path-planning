import os
import shutil

from wcci_conference_project.ai_modules.generate_real_dataset import generate_dataset, resolve_dem_path
from wcci_conference_project.ai_modules.train_risk_model import train

DATASET_NAME = "risk_dataset_glos.csv"
SAVE_STEM = "risk_dataset_glos"
SCENARIO_IDS = ["S1_Base", "S2_Dense", "S3_Long"]
NUM_SAMPLES = int(os.environ.get("DNN_TRE_NUM_SAMPLES", "18000"))
EPOCHS = int(os.environ.get("DNN_TRE_EPOCHS", "220"))
LEARNING_RATE = float(os.environ.get("DNN_TRE_LR", "0.0008"))
BATCH_SIZE = int(os.environ.get("DNN_TRE_BATCH_SIZE", "224"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("DNN_TRE_PATIENCE", "30"))
FOCUS_PROB = float(os.environ.get("DNN_TRE_FOCUS_PROB", "0.85"))


def run_pipeline():
    dem_path = resolve_dem_path()
    print(f"DEM Path Check: {dem_path}")
    if dem_path is None:
        print("DEM bulunamadi. Pipeline durduruldu.")
        return

    print("Starting DNN-TRE dataset generation...")
    generate_dataset(
        num_samples=NUM_SAMPLES,
        scenario_ids=SCENARIO_IDS,
        save_stem=SAVE_STEM,
        dem_path=dem_path,
        focus_prob=FOCUS_PROB,
    )

    print("Starting DNN-TRE model training...")
    train(
        dataset_name=DATASET_NAME,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    drive_mount = "/content/drive/My Drive"
    if os.path.exists(drive_mount):
        dest_dir = os.path.join(drive_mount, "msc")
        if not os.path.exists(dest_dir):
            dest_dir = drive_mount

        for filename in [
            "neural_risk.pth",
            "neural_risk_meta.json",
            f"{SAVE_STEM}.csv",
            f"{SAVE_STEM}_meta.json",
        ]:
            src = os.path.join(os.path.dirname(__file__), filename)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dest_dir, filename))
                print(f"Backed up to Drive: {os.path.join(dest_dir, filename)}")


if __name__ == "__main__":
    run_pipeline()
