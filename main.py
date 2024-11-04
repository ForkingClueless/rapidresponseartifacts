from run import run_with_pm2, generate_behaviors
import json


MODELS = [
    "meta-llama/Llama-3-8b-chat-hf",
    "gpt-4o-2024-08-06",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

ATTACKS = ['pair', 'renellm', 'crescendo', 'skeleton_key', 'msj', 'cipher']


MODES = [
    ("train", "iid"),
    ("test_iid", "iid"),
    ("test_ood", "ood"),
]

# run_with_pm2(
#     "renellm_iid",
#     MODELS[1],
#     "test_iid",
# )
run_with_pm2(
    "skeleton_key_ood",
    MODELS,
    "test_ood",
)
#
#
# exit()
# for dataset, suffix in MODES:
#     for a in ATTACKS[5:]:
#         # for m in MODELS:
#         atk_type = f"{a}_{suffix}"
#         # if "gpt" in m and dataset == "test_iid":
#         #     continue
#         run_with_pm2(
#             atk_type,
#             MODELS,
#             dataset
#         )