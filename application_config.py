import os
import sys
import log

#TODO: Magari sta roba meglio buttarli in un file .config con qualche formato standardizzato tipo yaml o che ne so.

#APPPLICATION PATHS --------------------------------
MAIN_PATH = ""
IMGS_PATH = ""
MESH_FILE = ""
METASHAPE_FILE = ""

PSEUDOLABEL_OUTPUT_PATH = ""
RITM_OUTPUT_PATH = ""
#---------------------------------------------------

#RITM CONFIGS -------------------------------------- #In realta' fa config di altra roba non solo del ritm
MAX_REGION_AREA = 300

GENERATE_WITH_ORIGINAL_BG = True
WORKERS_AMOUNT = 4

P_POINTS_AMOUNT = 4
N_POINTS_AMOUNT = 8

EROSION_AMOUNT = 10 #In pixels

RITM_ITER_AMOUNT = 10 #Quante volte RITM dovra' essere eseguito sulla stessa regione.
VOTE_THRESHOLD = 8 #Quanti voti servono per ammettere un punto nella segmentazione finale

DEBUG_PRINT_REGIONMAP = False
DEBUG_PRINT_ALL_RITM_SEGMENTATIONS = False
#---------------------------------------------------

def init(): #TODO: In un mondo ideale quando main.py e image_test.py verranno fusi questa funzione viene chiamata solo nell'entry point dell'applicazione 1 volta.
    load_filepaths()

    global PSEUDOLABEL_OUTPUT_PATH, RITM_OUTPUT_PATH
    PSEUDOLABEL_OUTPUT_PATH = os.path.join(MAIN_PATH, "pseudo_label_output")
    RITM_OUTPUT_PATH = os.path.join(MAIN_PATH, "ritm_output")

    if not os.path.isdir(PSEUDOLABEL_OUTPUT_PATH):
        os.mkdir(PSEUDOLABEL_OUTPUT_PATH)

    if not os.path.isdir(RITM_OUTPUT_PATH):
        os.mkdir(RITM_OUTPUT_PATH)

    log.print_info(f"Pseudo label output path: {PSEUDOLABEL_OUTPUT_PATH}")
    log.print_info(f"Ritm output path: {RITM_OUTPUT_PATH}")

def load_filepaths():
    """
        Try to read the filepaths from sys.argv or last.txt.\n
        Output:
        - main_path
        - imgs_path
        - mesh_name
        - metashape_file
    """
    global MAIN_PATH, IMGS_PATH, MESH_FILE, METASHAPE_FILE

    if len(sys.argv) == 5:
        MAIN_PATH = sys.argv[1]
        IMGS_PATH = sys.argv[2]
        MESH_FILE = sys.argv[3]
        METASHAPE_FILE = sys.argv[4]
    else:
        with open("last.txt", "r") as f:
            lines = f.read().splitlines()
            if len(lines) >= 4:
                MAIN_PATH = lines[0]
                IMGS_PATH = lines[1]
                MESH_FILE = lines[2]
                METASHAPE_FILE = lines[3]
            else:
                print("[ERROR] last.txt does not contain enough lines.")

    log.print_info(f"Main path: {MAIN_PATH}")
    log.print_info(f"Images path: {IMGS_PATH}")
    log.print_info(f"Mesh: {MESH_FILE}")
    log.print_info(f"Metashape file: {METASHAPE_FILE}\n")