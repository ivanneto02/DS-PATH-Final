from dotenv import load_dotenv
import os

load_dotenv()

# mysql config
mysql_username = os.environ["_MYSQL_USERNAME"]
mysql_password = os.environ["_MYSQL_PASSWORD"]
mysql_port = os.environ["_MYSQL_PORT"]
mysql_host = os.environ["_MYSQL_HOST"]
mysql_database = os.environ["_MYSQL_DB"]

# Here include which CUIs to graph relationships with.
CUIs = ["C2355964", "C0714714", "C2356005"]#, "C2937430", "C2356007", "C2356018", "C2356019", "C2356020", "C2356021", "C2356029", "C2356031", "C2356046", "C2356047", "C2356048", "C2356049", "C2356057", "C2356067", "C2356069", "C2356070", "C2937432", "C2937433", "C2356087", "C2356088", "C2356089", "C2356090", "C2356091", "C2356095", "C2356096", "C2356097", "C2356098", "C2356099", "C2356101", "C2937434", "C2937435", "C2356120", "C2356121", "C2356122", "C2356129", "C2356131", "C2356136", "C2356137", "C2356138", "C2356158", "C2356161", "C2356164", "C2356192", "C2356193", "C2356194", "C2356195", "C2356214", "C2356249", "C2356250", "C2356251", "C2356268", "C0712406", "C2356282", "C2356283", "C2356287", "C2356288", "C2356289", "C2356290", "C2356291", "C2356292", "C2356294", "C1329481", "C1329300", "C2356304", "C1169549", "C2356307", "C2344119", "C2356372", "C0710716", "C2356378", "C2356540", "C2356546", "C2356547", "C2356548", "C2356560", "C2356562", "C2356575", "C2356577", "C2356603", "C2356604", "C2356605", "C2356606", "C2356651", "C2356652", "C2356653", "C2356654", "C2356674"]

# Save config
image_save_path = "./saves/graphs/"

DATA_PATH = os.environ["DATA_PATH"]
FULL_TABLE_IN_FILE = os.environ["FULL_TABLE_IN_FILE"]
FULL_TABLE_OUT_FILE = os.environ["FULL_TABLE_OUT_FILE"]
BARE_TABLE_IN_FILE = os.environ["BARE_TABLE_IN_FILE"]
BARE_TABLE_OUT_FILE = os.environ["BARE_TABLE_OUT_FILE"]
BARE_TABLE_OUT_FILE_STEP_1 = os.environ["BARE_TABLE_OUT_FILE_STEP_1"]
BARE_TABLE_OUT_FILE_STEP_2 = os.environ["BARE_TABLE_OUT_FILE_STEP_2"]
BARE_TABLE_OUT_FILE_STEP_3 = os.environ["BARE_TABLE_OUT_FILE_STEP_3"]
NFILES = int(os.environ["N_FILES"]) if "None" not in os.environ["N_FILES"] else None
FEATURES = int(os.environ["FEATURES"])

MYSQL_USERNAME = os.environ["_MYSQL_USERNAME"]
MYSQL_PASSWORD = os.environ["_MYSQL_PASSWORD"]
MYSQL_PORT = os.environ["_MYSQL_PORT"]
MYSQL_HOST = os.environ["_MYSQL_HOST"]
MYSQL_DATABASE = os.environ["_MYSQL_DB"]

CONCEPT_TABLES_SAVE_PATH = os.environ["_CONCEPT_TABLES_SAVE_PATH"]
RELATION_TABLES_SAVE_PATH = os.environ["_RELATION_TABLES_SAVE_PATH"]
FULL_TABLE_SAVE_PATH = os.environ["_FULL_TABLES_SAVE_PATH"]
BARE_TABLE_SAVE_PATH = FULL_TABLE_SAVE_PATH
IMAGES_SAVE_PATH = os.environ["_IMAGES_SAVE_PATH"]

PATH_TO_DATA = os.environ["PATH_TO_DATA"]
DATA_FILE = os.environ["DATA_FILE"]
N_ROWS = int(os.environ["N_ROWS"]) if "none" not in os.environ["N_ROWS"].lower() else None # use "None" in .env to indicate all rows

# Data parameters
DATA_PATH = "C:\\Users\\ivana\\Desktop\\Documents\\Research\\UCR\\DS-PATH\\working_dir\\data"
DATA_FILE = "scraped_preprocessed_fulltext_2.csv"
NROWS = 100
FEATURES = 300
SEP_COLUMN = 7
TARGET_COLUMN = "concept_type"
CLASSES = 2

# Model parameters
TFIDF_MODEL_NAME = "TFIDFModel"

# must add up to 1
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

# hyperparameters
EPOCHS = 20
BATCH_SIZE = 128