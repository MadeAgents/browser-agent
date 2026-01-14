import json
import os
import shutil
import argparse
from agent import BroswerAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


this_dir = os.path.dirname(__file__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Run experiments over multiple tasks.")
    parser.add_argument("--task", type=str, required=True, help="Task group name, e.g. cross, abc, etc.")
    parser.add_argument("--task_ids", type=int, nargs="+", required=False, help="Task id(s), e.g. 554 or 554 557 558, etc.")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name to be used in result path.")
    parser.add_argument("--rerun", action="store_true", help="Remove existing results and rerun all tasks.")
    parser.add_argument("--retry", action="store_true", help="Retry only failed tasks.")

    # Other parameters remain unchanged or add as needed
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--visual_effects", type=str2bool, default=True)
    parser.add_argument("--use_html", type=str2bool, default=False)
    parser.add_argument("--use_axtree", type=str2bool, default=False)
    parser.add_argument("--use_screenshot", type=str2bool, default=True)
    parser.add_argument("--use_som", type=str2bool, default=True, help="Whether to save SOM (Self-Organizing Map) in the results.")
    parser.add_argument("--mode", type=str, default="bid", help="Mode of the experiment, e.g. bid, demo, etc.")
    parser.add_argument("--tips", type=str2bool, default=False, help="Whether to use tips in the agent's observation space.")
    parser.add_argument("--headless", type=str2bool, default=True, help="Whether to run the experiment in headless mode.")
    parser.add_argument("--use_full_action_history", type=str2bool, default=True, help="Whether to use full action history in the agent's observation space.")
    return parser.parse_args()


def setup_env_vars():
    from dotenv import load_dotenv
    load_dotenv()


def run_single_task(task_id, task_name, base_result_dir, websites, args):
    # Construct the final result directory
    results_dir = os.path.join(base_result_dir, task_name)

    if os.path.exists(results_dir):
        print("results_dir: ", results_dir)
        if args.retry:  # Retry only unsuccessful tasks
            if os.path.exists(os.path.join(results_dir, "summary_info.json")):
                with open(os.path.join(results_dir, "summary_info.json"), "r") as f:
                    summary_info = json.load(f)
                    if summary_info["cum_reward"] > 0.0:
                        print(f"⏩ Directory {results_dir} already exists and cum_reward > 0.0. Skip.")
                        return
        if not args.rerun and not args.retry:
            print(f"⏩ Directory {results_dir} already exists. Skip.")
            return
        else:
            print(f"🔁 Clearing directory {results_dir} due to rerun.")
            shutil.rmtree(results_dir)

    print(f"Running task {task_name} ...")

    nav_multipages = len(websites) > 1
    if args.tips:
        tips_path = [os.path.join(this_dir, "tips", f"{args.task}.txt")]
        if len(websites) > 1:
            for website in websites:
                path = os.path.join(this_dir, "tips", f"{website}.txt")
                if os.path.exists(path):
                    tips_path.append(path)
    else:
        tips_path = None
    agent_args = BroswerAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
        mode = args.mode,
        use_som=args.use_som,
        nav_multipages=nav_multipages,
        tips_path=tips_path,
        task_type=args.task,
        use_full_action_history=args.use_full_action_history,
    )

    max_steps = 40 if len(websites) > 1 else 20
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=max_steps,
        headless=args.headless,
        # slow_mo=30,
    )

    if task_name == "openended":
        agent_args.chat_mode = True
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": "https://www.google.com"}

    exp_args = ExpArgs(env_args=env_args, agent_args=agent_args, save_som=True, save_json=True)
    exp_args.prepare(results_dir)
    exp_args.run()

    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()
    print(f"\nResults for task {task_name}:")
    for k, v in exp_record.items():
        print(f"{k}: {v}")


def main():
    args = get_args()
    setup_env_vars()
    print(f"🔍 Args: {args}")

    # Load the corresponding TASK_IDS list according to different task names
    # This example shows the task id list for cross, you can modify or dynamically pass in as needed

    task_id2sites = json.load(open(os.path.join(this_dir, "webarena/task_id2sites.json")))
    sites2task_id = json.load(open(os.path.join(this_dir, "webarena/sites2task_id.json")))
    if args.task not in sites2task_id:
        raise ValueError(f"Unknown task group: {args.task}, available task groups: {list(sites2task_id.keys())}")

    TASK_IDS = sites2task_id[args.task]
    if args.task_ids:
        for task_id in args.task_ids:
            if task_id not in TASK_IDS:
                raise ValueError(f"Unknown task id: {task_id}")
        TASK_IDS = args.task_ids

    base_result_dir = os.path.join("results", "webarena", args.task, args.exp)

    if args.rerun and os.path.exists(base_result_dir):
        print(f"🔁 Clearing directory {base_result_dir}")
        shutil.rmtree(base_result_dir)

    for task_id in TASK_IDS:
        websites = task_id2sites[str(task_id)]
        task_name = f"webarena.{task_id}"
        run_single_task(task_id, task_name, base_result_dir, websites, args)
        # break
        # input("Press Enter to continue...")


if __name__ == "__main__":
    main()
