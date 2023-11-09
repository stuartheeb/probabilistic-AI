from ray import train, tune
import json
import subprocess
import os


def objective(params):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(r'C:\Users\nicos\eth-probabilistic-AI\projects\task2\params.json', 'w') as f:
        json.dump(params, f)

    # {'swag_epochs': 25, 'swag_lr': 0.012222562545145955, 'bma_samples': 20, 'prediction_threshold': 0.7083039219612638, 'matrix_rank': 12}
    print("json file should be there  ", params)
    p = subprocess.Popen(["powershell.exe", r"C:\Users\nicos\eth-probabilistic-AI\projects\task2\docker.ps1"], stdout=subprocess.PIPE)
    p_out, p_err = p.communicate()
    print(p_out)
    print(str(p_out))
    print("error is:  ", p_err)
    if "worse" in str(p_out):
        cost = str(p_out).split("penalized PUBLIC cost")[1].split("thus worse")[0].replace('is', '').replace(' ', '')
    else:
        cost = str(p_out).split("penalized PUBLIC cost is ")[1][:5]
    return {"cost": float(cost)}

# SWAG_EPOCHS = 30
# SWAG_LR = 0.035
# BMA_SAMPLES = 30
# PREDICTION_THRESHOLD = 0.71
# MATRIX_RANK = 15

search_space = {
    "swag_epochs": tune.grid_search([25, 30, 35, 40, 50]),
    "swag_lr": tune.grid_search([0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05]),
    "bma_samples": tune.grid_search([20, 30, 40, 50]),
    "prediction_threshold": tune.grid_search([0.6, 0.65, 0.7, 0.75, 0.8]),
    "matrix_rank": tune.grid_search([12, 15, 18])
}


tuner = tune.Tuner(tune.with_resources(objective, resources={"gpu": 1, "cpu" : 12}), param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="cost", mode="min").config)
