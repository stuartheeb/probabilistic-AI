from ray import train, tune
import json
import subprocess
import os
import time
import shutil

from ray.tune.search.bayesopt import BayesOptSearch


def objective(params):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params["swag_epochs"], params["bma_samples"], params["matrix_rank"] = int(params["swag_epochs"]), int(params["bma_samples"]), int(params["matrix_rank"])
    with open(r'C:\Users\nicos\eth-probabilistic-AI\projects\task2\params.json', 'w') as f:
        json.dump(params, f)

    time.sleep(5)  # to be sure it writes it

    # {'swag_epochs': 25, 'swag_lr': 0.012222562545145955, 'bma_samples': 20, 'prediction_threshold': 0.7083039219612638, 'matrix_rank': 12}
    print("json file should be there  ", params)
    p = subprocess.Popen(["powershell.exe", r"C:\Users\nicos\eth-probabilistic-AI\projects\task2\docker.ps1"],
                         stdout=subprocess.PIPE)
    p_out, p_err = p.communicate()
    print(p_out)
    print(str(p_out))
    print("error is:  ", p_err)
    if "worse" in str(p_out):
        cost = str(p_out).split("penalized PUBLIC cost")[1].split("thus worse")[0].replace('is', '').replace(' ', '')
    else:
        cost = str(p_out).split("penalized PUBLIC cost is ")[1][:5]

    # copy resultsfile to old ones if < 0.82
    if float(cost) < 0.82:
        src = r'C:\Users\nicos\eth-probabilistic-AI\projects\task2\results_check.byte'
        dst = fr'C:\Users\nicos\eth-probabilistic-AI\projects\task2\old_checkfiles\results_check_' + str(cost) + ".byte"
        shutil.copy(src, dst)
        print(f"copied the file with cost {float(cost)} to old_checkfiles!")
    return {"cost": float(cost)}


algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})

# {'swag_epochs': 29, 'swag_lr': 0.010387378514881005, 'bma_samples': 32, 'prediction_threshold': 0.7297257824306465, 'matrix_rank':16}
search_space = {
    "swag_epochs": tune.uniform(25, 35),
    "swag_lr": tune.uniform(0.008, 0.012),
    "bma_samples": tune.uniform(28, 34),
    "prediction_threshold": tune.uniform(0.71, 0.76),
    "matrix_rank": tune.uniform(15, 20)
}

tuner = tune.Tuner(tune.with_resources(objective, resources={"gpu": 1, "cpu": 12}),
                   param_space=search_space,
                   tune_config=tune.TuneConfig(
                       metric="cost",
                       mode="min",
                       search_alg=algo,
                       num_samples=50
                   ))
results = tuner.fit()
print(results.get_best_result().config)
