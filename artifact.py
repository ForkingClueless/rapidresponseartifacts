from utils.judge import is_jailbroken
from utils.parallel import parallel_map


def find_root_query(instance) -> str:
    """
    Recursively find the root query by traversing the
     parent chain.
    """
    if not instance.parents:
        return instance.query
    return find_root_query(instance.parents[0])


def attack_artifacts(
    attacker,
    attack_method,
    target_model,
    dataset_name,
):

    import os.path as p
    import os

    raw_dir = p.join(p.dirname(p.abspath(__file__)), "raw")
    # save to raw_dir / attack_method / target_model / dataset_name

    path = p.join(raw_dir, attack_method, target_model)
    os.makedirs(path, exist_ok=True)
    
    raw_path = p.join(path, f"{dataset_name}.json")

    if p.exists(raw_path):
        print(f"{raw_path} already exists!")
        return

    to_save = []
    while True:
        # all methods are 1 to 1 for input and output dataset
        # except for cipher -- cipher is 1 to N; in this case
        # just randomly pick one of the N
        dataset_length = len(attacker.jailbreak_datasets)
        attacker.attack()

        results = attacker.attack_results

        # for tests sets -- try to get one for each behavior
        # if training set -- try to get one for each behavior, but duplicates are okay
        # to hit 25 working jailbreaks
        if len(results) > dataset_length and "train" not in dataset_name:
            # need determinism
            import random

            rng = random.Random(1337)
            step = len(results) // dataset_length
            assert len(results) % dataset_length == 0
            results = [
                results[rng.randrange(i, i + step)]
                for i in range(0, len(results), step)
            ]

        new_results = []
        for i in results:
            prompt = i.jailbreak_prompt
            if "{query}" in prompt:
                prompt = prompt.replace("{query}", i.query)
            if "{encoded_query}" in prompt:
                prompt = prompt.replace("{encoded_query}", i.encoded_query)

            query = find_root_query(i)

            responses = i.target_responses
            encoded_responses = None

            multi_turn_convo = None

            try:
                encoded_responses = i.encoded_target_responses
            except AttributeError:
                pass

            try:
                multi_turn_convo = i.multi_turn_convo
            except AttributeError:
                pass

            if len(responses) != 1:
                print("weird responses: ", responses)

            new_results.append(
                dict(
                    attack_method=attack_method,
                    target_model=target_model,
                    prompt=prompt,
                    query=query,
                    responses=responses,
                    encoded_responses=encoded_responses,
                    multi_turn_convo=multi_turn_convo,
                )
            )

        print("checking is_jailbroken")
        is_jailbrokens = parallel_map(
            is_jailbroken,
            [i["query"] for i in new_results],
            [i["responses"][0] for i in new_results],
            use_tqdm=True,
        )

        openai_messages = []
        for i in new_results:
            if i["multi_turn_convo"]:
                openai_messages.append(i["multi_turn_convo"])
            else:
                openai_messages.append([{"role": "user", "content": i["prompt"]}])

        print("checking blocked_by_guard")
        blocked_by_guards = parallel_map(
            attacker.target_model.blocked_by_guard,
            messages=openai_messages,
            concurrency=5,
            use_tqdm=True,
        )

        for d, is_jb, is_blocked in zip(new_results, is_jailbrokens, blocked_by_guards):
            d["jailbreaks_policy"] = is_jb
            d["jailbreaks_guard"] = not is_blocked

        to_save.extend(new_results)

        bypassed_guard = sum(d["jailbreaks_guard"] for d in to_save)
        bypassed_all = sum(
            d["jailbreaks_policy"] & d["jailbreaks_guard"] for d in to_save
        )
        print(f"bypassed_guard: {bypassed_guard}/{len(to_save)}")
        print(f"bypassed_all: {bypassed_all}/{len(to_save)}")

        if "train" not in dataset_name:
            break
        else:
            # if no jailbreaks (MSJ for gpt-4o), running again won't help at all
            if bypassed_all >= 25 or bypassed_all == 0:
                break
            else:
                print("not enough jailbreaks found, repeating search...")

    import json
    with open(raw_path, "w") as f:
        json.dump(to_save, f, indent=4)
