import os
import pandas as pd
import logging
from data_transformer import transform_datasets

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    notebooks_dir = os.path.join(base_dir, 'small_notebooks')

    parameter_file = os.path.join(base_dir, 'Parameter_Values.json')

    notebooks_to_process = [
        'kaggle__aadarshvelu__spaceship-titanic-tensorflow-80/spaceship-titanic-tensorflow-80.ipynb',
        'kaggle__aaron95629__spaceship-titanic/spaceship-titanic.ipynb',
        'kaggle__abdullahalbunni__spaceship-titanic-with-randomforestclassifier/spaceship-titanic-with-randomforestclassifier.ipynb',
        'kaggle__abraamsaid__spaceship-titanic-eda-predictions/spaceship-titanic-eda-predictions.ipynb',
        'kaggle__adends__spaceship-titanic-competition-with-ensemble-models/spaceship-titanic-competition-with-ensemble-models.ipynb',
        'kaggle__adhamad0__spaceship-titanic/spaceship-titanic.ipynb',
        'kaggle__aestroe__spaceship-titanic-classification/spaceship-titanic-classification.ipynb',
        'kaggle__ahmedanwar89__spaceship-titanic-ml/spaceship-titanic-ml.ipynb',
        'kaggle__ahmedgaitani__spaceship-titanic-code/spaceship-titanic-code.ipynb',
        'kaggle__akankshasolanki123__spaceship-titanic/spaceship-titanic.ipynb'
    ]
    
    try:
        train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
        logger.info("Successfully loaded original datasets")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return

    for notebook_relative_path in notebooks_to_process:
        notebook_path = os.path.join(notebooks_dir, notebook_relative_path)
        
        if not os.path.exists(notebook_path):
            logger.error(f"Notebook not found: {notebook_path}")
            continue
            
        logger.info(f"\nProcessing notebook: {notebook_path}")
        
        try:
            transformed_train, transformed_test = transform_datasets(
                notebook_path=notebook_path,
                train_df=train_df,
                test_df=test_df,
                parameter_file=parameter_file
            )

            notebook_dir = os.path.dirname(notebook_path)
            transformed_train.to_csv(
                os.path.join(notebook_dir, 'transformed_train.csv'),
                index=False
            )
            transformed_test.to_csv(
                os.path.join(notebook_dir, 'transformed_test.csv'),
                index=False
            )

            logger.info(f"Saved transformed datasets for {notebook_relative_path}")

        except Exception as e:
            logger.error(f"Error processing {notebook_relative_path}: {e}")
            continue

    logger.info("\nCompleted processing all notebooks")

if __name__ == "__main__":
    main()