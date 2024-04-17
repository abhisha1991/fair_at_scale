# DO NOT CHANGE THIS
PARENT_PATH = "/gdrive/MyDrive/FairInfluenceMaximization/"
DATA_PATH = PARENT_PATH + "data/"
DATA_PATH_UNZIPPED = DATA_PATH + "Data/"
CODE_PATH = PARENT_PATH + "code/"
DATA_INIT_PATH = DATA_PATH_UNZIPPED + "Weibo/Init_Data/"
DATA_FPS_PATH = DATA_PATH_UNZIPPED + "Weibo/FPS/"
DATA_FAC_PATH = DATA_PATH_UNZIPPED + "Weibo/FAC/"
RUN_ID = "Full-attempt_2024-04-07_synethic_gender_x_pol_affln_noisy"  # this can change

DATA_OUTPUT_PATH = DATA_PATH_UNZIPPED + f"Weibo/Output_{RUN_ID}/"
DATA_EMBEDDINGS_PATH = DATA_OUTPUT_PATH + "Embeddings/"
DATA_SEEDS_PATH = DATA_OUTPUT_PATH + "Seeds/"
DATA_SEEDS_SPREADING_PATH = DATA_OUTPUT_PATH + "Spreading/"
DATA_ZIP = "Data.zip"
OPT_DATA = "/opt/data/"

# define main control parameters
INPUT_FN = "weibo"
ATTRIBUTE = "age_x_pol_affln_with_noise"
SAMPLING_PERC = 120
LEARNING_RATE = 0.1
N_EPOCHS = 1
EMBEDDING_SIZE = 50
NUM_NEG_SAMPLES = 10

# file paths
TRAIN_CASCADES_FULL = DATA_INIT_PATH + "train_cascades.txt"
TEST_CASCADES = DATA_INIT_PATH + "test_cascades.txt"
GENDER_ATTRIBUTE_CSV = DATA_INIT_PATH + f"profile_{ATTRIBUTE}.csv"
TRAIN_SET_FILE = DATA_OUTPUT_PATH + f"train_set_{RUN_ID}.txt"
TRAIN_CASCADES_SMALL = DATA_OUTPUT_PATH + f"train_cascades_{RUN_ID}.txt"

# overrides if any (temp)
SAMPLING_PERC = 5

REPO_FAIR_IM_PATH = CODE_PATH + "fair_at_scale/"
REPO_IMINFECTOR_PATH = CODE_PATH + "fair_at_scale/IMINFECTOR/"
