RAW_FILE = "houseprice.csv"
OUTPUT_SCALER_PATH = "scaler.pkl"
OUTPUT_MODEL = "model.pkl"

TARGET_VARIABLE = "SalePrice"
SPLIT_PERCENTAGE=0.1
RARE_PERCENTAGE=0.01

MISSING_CATEGORICAL=['Alley',
 'MasVnrType',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Electrical',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'Fence',
 'MiscFeature']

MISSING_NUMERICAL = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

REPLACE_CATEGORICAL = "Missing"

REPLACE_NUMERICAL = "mode"

TEMP_VARIABLES = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']

LOG_VARIABLES = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

EXCLUDE_VARIABLES = ["Id","SalePrice"]

FROM_COLUMN = 'YrSold'

ENCODING_INFREQUENT = "Rare"

SELECTED_FEATURES = ['MSSubClass',
 'MSZoning',
 'Neighborhood',
 'OverallQual',
 'OverallCond',
 'YearRemodAdd',
 'RoofStyle',
 'MasVnrType',
 'BsmtQual',
 'BsmtExposure',
 'HeatingQC',
 'CentralAir',
 '1stFlrSF',
 'GrLivArea',
 'BsmtFullBath',
 'KitchenQual',
 'Fireplaces',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageCars',
 'PavedDrive']
