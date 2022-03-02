from glob import escape
import os
import numpy as np
import sys

def parse_dir(dir_name, prefix, suffix):
    eval_log_list = {}
    eval_best_list = {}
    for fname in os.listdir(dir_name):
        if not fname.startswith(prefix):
            continue
        print(fname)
        if (fname.startswith("base_postln_") or fname.startswith("base_preln_") or fname.startswith("large_postln_") or fname.startswith("large_preln_")) and fname.endswith(suffix):
            eval_log_path = "{0}/{1}".format(dir_name, fname)
            eval_log_list[eval_log_path] = {}
            if suffix == "":
                seed = int(fname.split("_")[-1])
            else:
                seed = int(fname.split("_")[-2])
            eval_best_list[seed] = {}
            with open(eval_log_path, "r") as in_file:
                all_checkpoint_num = set()
                for line in in_file:
                    if line.find("| INFO | __main__ | evaluating checkpoint file") != -1:
                        checkpoint_num = line.strip().split(".")[-2].split("checkpoint")[-1].strip("_")
                        all_checkpoint_num.add(checkpoint_num)
                    if line.find("| INFO | graphormer.tasks.graph_prediction | Loaded") != -1:
                        if line.find("| INFO | graphormer.tasks.graph_prediction | Loaded test") != -1:
                            split = "test"
                        elif line.find("| INFO | graphormer.tasks.graph_prediction | Loaded valid") != -1:
                            split = "valid"
                        if split not in eval_log_list[eval_log_path]:
                            eval_log_list[eval_log_path][split] = {}
                    if line.find("| INFO | __main__ | auc:") != -1:
                        auc = float(line.strip().split(": ")[-1])
                        eval_log_list[eval_log_path][split][checkpoint_num] = auc
                        print(split, auc)
            valid_best = "best"
            print(all_checkpoint_num)
            for checkpoint_num in np.sort(list(all_checkpoint_num)):
                print("{0} valid {1} test {2}".format(checkpoint_num, eval_log_list[eval_log_path]["valid"][checkpoint_num], eval_log_list[eval_log_path]["test"][checkpoint_num]))
                if eval_log_list[eval_log_path]["valid"][valid_best] < eval_log_list[eval_log_path]["valid"][checkpoint_num]:
                    valid_best = checkpoint_num
            eval_best_list[seed]["valid"] = eval_log_list[eval_log_path]["valid"][valid_best]
            eval_best_list[seed]["test"] = eval_log_list[eval_log_path]["test"][valid_best]
            eval_best_list[seed]["valid_best_loss"] = eval_log_list[eval_log_path]["valid"]["best"]
            eval_best_list[seed]["test_best_loss"] = eval_log_list[eval_log_path]["test"]["best"]
    values = []
    for seed in np.sort(list(eval_best_list.keys())):
        values.append([eval_best_list[seed]["valid"], eval_best_list[seed]["test"], eval_best_list[seed]["valid_best_loss"], eval_best_list[seed]["test_best_loss"]])
    values = np.array(values)
    mean = values.mean(axis=0)
    print(values)
    std = values.std(axis=0)
    print("valid-mean {0} test-mean {1} valid-best-loss-mean {2} test-best-loss-mean {3}".format(mean[0], mean[1], mean[2], mean[3]))
    print("valid-std {0} test-std {1} valid-best-loss-std {2} test-best-loss-std {3}".format(std[0], std[1], std[2], std[3]))
    return eval_log_list, eval_best_list

if __name__ == "__main__":
    if len(sys.argv) == 3:
        parse_dir(sys.argv[1], sys.argv[2], "")
    elif len(sys.argv) == 4:
        parse_dir(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("usage: python parse_results.py <dir_name> <prefix> [suffix]")
        exit(-1)
