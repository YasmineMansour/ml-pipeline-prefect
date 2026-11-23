import argparse
import os

from model_pipeline import prepare_data, train_model, evaluate_model

parser = argparse.ArgumentParser(
    description="Pipeline ML - Réadmission des patients diabétiques"
)
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--evaluate", action="store_true")

args = parser.parse_args()

if args.prepare:
    X_train, X_test, y_train, y_test, _ = prepare_data()
    print(
        f"Données prêtes : train={X_train.shape}, "
        f"test={X_test.shape}, "
        f"% réadmission train={y_train.mean():.3f}"
    )

if args.train:
    if not os.path.exists("outputs/prepared_data.joblib"):
        print("Préparation des données...")
        prepare_data()
    train_model()

if args.evaluate:
    if not os.path.exists("outputs/model.joblib"):
        parser.error("Modèle introuvable. Lance d'abord python3 main.py --train")
    evaluate_model()
