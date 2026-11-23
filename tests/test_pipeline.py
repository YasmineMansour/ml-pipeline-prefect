import model_pipeline


def test_fonctions_presentes():
    for nom in ["prepare_data", "train_model", "evaluate_model",
                "save_model", "load_model"]:
        assert hasattr(model_pipeline, nom)
