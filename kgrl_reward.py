from utils import gsm8k_format_reward, gsm8k_acc_reward, get_kgrl_ie_config, get_kgrl_mode_config, \
    mmedbench_format_reward, mmedbench_acc_reward, kgrl_light_reward, kgrl_dark_reward, result_string_parse



def normalize_reward(scores = None, score_weight = None):
    weights = [value for key, value in score_weight.items() if key in scores.keys()]
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0
    weighted_scores = [scores[key] * score_weight[key] for key in scores.keys()]
    return sum(weighted_scores)/total_weight


def compute_score(data_source = None, solution_str = None, ground_truth = None, extra_info = None):
    ground_truth_result = ground_truth 
    if 'gsm8k' in data_source.lower():
        scores = {
            'format': gsm8k_format_reward(solution_str),
            'accuracy': gsm8k_acc_reward(solution_str, ground_truth_result)
        }
        score_weight = {
            'format': 1,
            'accuracy': 4
        }
        reward = normalize_reward(scores, score_weight)
        return reward
    
    if 'mmedbench' in data_source.lower():
        kgrl_mode = get_kgrl_mode_config()
        alpha_ = 'alpha' in kgrl_mode
        beta_ = 'beta' in kgrl_mode
        full_ = 'full' in kgrl_mode
        light_ = 'light' in kgrl_mode
        vanilla_ = 'vanilla' in kgrl_mode
        weighted_ = 'weighted' in kgrl_mode
        strict_ = 'strict' in kgrl_mode         # 多选的reward是否严格按照所有选项正确才给�?        check_k = 0 if strict_ else 1           # 若不严格，则设为1，计算答案和ground truth �?F1分数

        alpha, beta = 2, 2
        try:
            alpha = int(kgrl_mode.split('alpha')[-1].split('_')[0]) if alpha_ else 2
        except (IndexError, ValueError):
            pass
        try:
            beta = int(kgrl_mode.split('beta')[-1].split('_')[0]) if beta_ else 2
        except (IndexError, ValueError):
            pass

        scores = {
                'format': mmedbench_format_reward(solution_str),
                'accuracy': mmedbench_acc_reward(solution_str, ground_truth_result, check_k=check_k)
            }
        score_weight = {
                'format': alpha,
                'accuracy': 10 - alpha
            }
        
        if vanilla_:
            reward = normalize_reward(scores, score_weight)
            return reward
        
        if light_:
            # print(extra_info)
            # entity_info = result_string_parse(solution_str, tag='ENTITY INFO')
            # print(entity_info)
            entity_temperature = 1
            try:
                entity_temperature = int(kgrl_mode.split('etmp')[-1].split('_')[0])
                entity_temperature = min(max(entity_temperature, 1), 5)
            except (IndexError, ValueError):
                pass
            entity_reward_weight = 0
            try:
                entity_reward_weight = int(kgrl_mode.split('eweight')[-1].split('_')[0])
                if entity_reward_weight != 0:
                    entity_reward_weight = min(max(entity_reward_weight, 1), score_weight['accuracy'] - 1)
            except (IndexError, ValueError):
                pass
            
            # if entity_score is None or len(entity_score) == 0:
            #     entity_score, entity_reward_weight = 0.0, 0.0
            # else:
            #     if weighted_:
            #         entity_score = sum(a * b for a, b in zip(entity_score, phrase_weight)) / (sum(phrase_weight) + 1e-5)
            #     else:
            #         entity_score = sum(entity_score) / (len(entity_score) + 1e-5)

            entity_score, use_entity_weight_ = kgrl_light_reward(
                solution_str, 
                extra_info.get('gold_entities', []), 
                weighted_=weighted_,
                entity_temp=entity_temperature,
            )
            if not use_entity_weight_:
                entity_score, entity_reward_weight = 0.0, 0
            else:
                if entity_reward_weight == 0: 
                    scores['accuracy'] *= entity_score
                else:
                    scores['entity'] = entity_score
                    score_weight['entity'] = entity_reward_weight
                    score_weight['accuracy'] = max(score_weight['accuracy'] - entity_reward_weight, 1)
        
        elif full_:
            # lambda_temperature �?0/1/2/3/4分别代表 0�?/4�?/2�?/4�? 的温度系数。lambda=1表示最注重gold
            try:
                lambda_temperature = int(kgrl_mode.split('lambda')[-1].split('_')[0])
            except IndexError:
                lambda_temperature = 4
            # entity_top_k �?代表使用等同于gold的实体数量，�?代表使用等同于option-gold的实体数量，2表示使用一半的option数量
            entity_low_top_k_signal = int(kgrl_mode.split('topk')[-1].split('_')[0])
            entity_low_top_k = 0
            num_high_entities = len(extra_info.get('gold_entities', []))
            num_low_entities = len(extra_info.get('extend_entities', []))

            if num_high_entities == 0 and num_low_entities == 0:
                # 尚未搜寻到任何知识片段时，按照accuracy给分，不考虑entity_score
                pass

            else:
                max_low_top_ratio = 0.2
                try:
                    max_low_top_ratio = float(kgrl_mode.split('topkr')[-1].split('_')[0])/100
                except (IndexError, ValueError):
                    pass

                entity_low_top_k = int(
                    num_low_entities * max_low_top_ratio * \
                    num_low_entities / (num_high_entities + num_low_entities + 1e-5)
                )

                entity_temperature = 2
                try:
                    entity_temperature = int(kgrl_mode.split('etmp')[-1].split('_')[0])
                    entity_temperature = min(max(entity_temperature, 1), 5)
                except (IndexError, ValueError):
                    pass
                
                if lambda_temperature != 0:
                    high_score, use_high_weight_ = kgrl_light_reward(
                        solution_str, 
                        extra_info.get('gold_entities', []), 
                        weighted_=False,
                        entity_temp=entity_temperature,
                    )
                else:
                    high_score, use_high_weight_ = 0, False

                if lambda_temperature != 4:
                    low_score, use_low_weight_ = kgrl_dark_reward(
                        solution_str, 
                        extra_info.get('extend_entities', []), 
                        weighted_=False,
                        entity_temp=entity_temperature,
                        top_k=entity_low_top_k
                    )
                else:
                    low_score, use_low_weight_ = 0, False

                entity_score = 1.0
                use_entity_weight_ = use_high_weight_ or use_low_weight_
                if use_entity_weight_:
                    if use_high_weight_:
                        entity_score *= high_score ** (lambda_temperature / 4)
                    if use_low_weight_:
                        entity_score *= low_score ** ((4 - lambda_temperature) / 4)
                    score_weight['accuracy'] = score_weight['accuracy'] * (1 - beta/10 + beta/10 * entity_score)
                    # entity_score = entity_score * (1 - beta/10 + beta/10 * scores['accuracy'])  # entity_score与accuracy_score进行融合，beta控制融合的程�?                    # score_weight['accuracy'] = 0
                    # scores['entity'] = entity_score
                    # score_weight['entity'] = 10 - alpha

                

            # scores['entity'] = entity_score
            # score_weight['entity'] = 2

        reward = normalize_reward(scores, score_weight)
        return reward
        
        # return 0.0
    
    # if 'qa' in data_source.lower():
    #     kgrl_mode = get_kgrl_mode_config()
    #     full_ = 'full' in kgrl_mode
    #     light_ = 'light' in kgrl_mode
    #     if full_:
    #         kgrl_ie_host, kgrl_ie_port = get_kgrl_ie_config()
    #         reward = 0
    #         print(f'''IE host: {kgrl_ie_host}, port: {kgrl_ie_port}''')
    #         return reward
    #     if light_:
    #         reward = 1
    #     return reward

