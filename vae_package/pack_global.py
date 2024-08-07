
class Object:
    pass

NJOBS_MULTIPROC = 8

RESULT_DIR = 'result/'

def_setting = Object()  # default setting
def_setting.d_model = 128
def_setting.d_latent = 128
def_setting.params = {
    'BATCH_SIZE': 128,
    'BETA_INIT': 0.05,
    'BETA': 0.05,
    'ANNEAL_START': 0
}