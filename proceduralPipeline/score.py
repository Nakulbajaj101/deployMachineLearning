from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from config import ( ENCODING_INFREQUENT,EXCLUDE_VARIABLES,  FROM_COLUMN,
                    LOG_VARIABLES, MISSING_CATEGORICAL, MISSING_NUMERICAL,
                    OUTPUT_MODEL, OUTPUT_SCALER_PATH, RAW_FILE, RARE_PERCENTAGE,
                    REPLACE_CATEGORICAL, REPLACE_NUMERICAL, SPLIT_PERCENTAGE,
                    TARGET_VARIABLE, TEMP_VARIABLES,SELECTED_FEATURES)
import numpy as np
import warnings
from preprocessing import Pipeline

warnings.simplefilter(action='ignore')

if __name__ == "__main__":

    pipeline = Pipeline(filepath=RAW_FILE,
                        split_percentage=SPLIT_PERCENTAGE,
                        rare_percentage = RARE_PERCENTAGE, 
                        scaler_output=OUTPUT_SCALER_PATH,
                        model_output=OUTPUT_MODEL,
                        target_variable=TARGET_VARIABLE,
                        missing_categorical=MISSING_CATEGORICAL,
                        missing_numerical=MISSING_NUMERICAL,
                        temp_variables=TEMP_VARIABLES,
                        log_variables=LOG_VARIABLES,
                        selected_features=SELECTED_FEATURES,
                        exclude_variables=EXCLUDE_VARIABLES,
                        from_column=FROM_COLUMN,
                        replace_categorical=REPLACE_CATEGORICAL,
                        replace_numerical=REPLACE_NUMERICAL,
                        encoding_infrequent=ENCODING_INFREQUENT)

    pipeline.process_data()
    pipeline.transform_data()
    pipeline.export_train_test()
    pipeline.train()
    pipeline.predict()
    pipeline.score()

    

