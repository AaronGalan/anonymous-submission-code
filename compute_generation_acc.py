import argparse
import os
from utils import gsm8k_acc_reward, read_parquet, mmedbench_acc_reward


# python compute_generation_acc.py --subset post3r --filename=MMedBench-English-grpo-Qwen2.5-3B-Instruct-vanilla-202508270959-global_step_600.parquet
# python compute_generation_acc.py --subset post3r --filename=MMedBench-English-grpo-Qwen2.5-3B-Instruct-light_weighted_eweight4_etmp2-202508281617-global_step_600.parquet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="The parquet file in ./data/dataset/checkpoint/model_name/ after generation script. Example: gsm8k-grpo-Qwen2.5-0.5B-Instruct-full-202508112323-global_step_6000.parquet",
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='',
        help="The subset of the dataset to evaluate. Example: 'post3r' for mmedbench."
    )
    return parser.parse_args()


def acc_reward(dataset: str, response: str, ground_truth: str) -> float:
    if 'gsm8k' in dataset.lower():
        return gsm8k_acc_reward(response, ground_truth)
    if 'mmedbench' in dataset.lower():
        return mmedbench_acc_reward(response, ground_truth)
    return gsm8k_acc_reward(response, ground_truth)


if __name__ == "__main__":
    args = parse_args()
    tags = args.filename.split('-')
    if 'Chinese' in tags[1] or 'English' in tags[1]:
        dataset, rl_algorithm, model_name = tags[0], tags[2], '-'.join(tags[3:7])
    else:
        dataset, rl_algorithm, model_name = tags[0], tags[1], '-'.join(tags[2:6])
    if args.subset != '':
        dataset = dataset + '/' + args.subset
    file_path = os.path.join("data", dataset, "checkpoint", model_name, args.filename)

    data = read_parquet(file_path)
    correct_count = 0
    total_count = 0
    for row in data:
        response = row.get("responses", [""])[0]
        ground_truth = row.get("reward_model", {}).get("ground_truth", "")
        if acc_reward(dataset, response, ground_truth) >= 0.99:
            correct_count += 1
        total_count += 1
        
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print('' + '='*20 + ' Result ' + '='*20)
    print(f"File: {file_path}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    print('='*48)
