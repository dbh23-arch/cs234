import re
import yaml, torch, gymnasium as gym, miniwob, numpy as np, json, os, sys
from collections import OrderedDict
from miniwob.action import ActionTypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ppo import PPOAgent, get_returns, calculate_advantage, update_policy, update_value
from utils.state_encoder import extract_dom_features

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs/default.yaml')))

tasks = [
    'click-button', 'click-link', 'click-option', 'click-dialog', 'click-dialog-2',
    'navigate-tree', 'click-checkboxes', 'email-inbox',
]
seeds = [42, 123, 456]

ppo_cfg = cfg['ppo']
num_batches = ppo_cfg['num_batches']
batch_size = ppo_cfg['batch_size']
max_ep_len = ppo_cfg['max_ep_len']
gamma = ppo_cfg['gamma']
eps_clip = ppo_cfg['eps_clip']
update_freq = ppo_cfg['update_freq']
entropy_coeff = ppo_cfg.get('entropy_coeff', 0.02)
lr = cfg['training']['lr']

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device:", device, flush=True)

def parse_dom(obs):
    elements = []
    for elem in obs.get('dom_elements', []):
        def to_float(v):
            return v.item() if hasattr(v, 'item') else float(v or 0)
        def to_int(v):
            if hasattr(v, "item"):
                v = v.item()
            try:
                return int(v)
            except Exception:
                return 0
        elements.append({'tag': str(elem.get('tag','div')), 'type': str(elem.get('type','')),
            'text': str(elem.get('text','')), 'value': str(elem.get('value','')),
            'left': to_float(elem.get('left',0)), 'top': to_float(elem.get('top',0)),
            'width': to_float(elem.get('width',0)), 'height': to_float(elem.get('height',0)),
            'visible': bool(elem.get('visible',True)), 'focused': bool(elem.get('focused',False)),
            'ref': to_int(elem.get("ref", 0))})
    return elements

def element_center(elem):
    cx = elem['left'] + elem['width'] / 2
    cy = elem['top'] + elem['height'] / 2
    cx = max(0.0, min(cx, 160.0))
    cy = max(0.0, min(cy, 210.0))
    return cx, cy

def is_input_like(elem):
    tag = str(elem.get("tag", "")).lower()
    typ = str(elem.get("type", "")).lower()
    if tag in ("input_text", "input_password", "input_search", "textarea"):
        return True
    if tag == "input":
        return typ in ("", "text", "password", "search", "email", "url", "tel")
    return tag.startswith("input_") and tag not in (
        "input_checkbox", "input_radio", "input_button", "input_submit",
        "input_reset", "input_file", "input_image", "input_hidden",
    )

def is_clickable(elem):
    tag = str(elem.get("tag", "")).lower()
    ref = int(elem.get("ref", 0))
    if ref <= 0:
        return False
    if tag.startswith("input_"):
        return True
    return tag in {
        "button", "a", "span", "label", "option", "select",
        "input", "li",
    }

def resolve_target_idx(elements, element_idx, predicate):
    if 0 <= element_idx < len(elements) and predicate(elements[element_idx]):
        return element_idx
    if not elements:
        return element_idx
    if 0 <= element_idx < len(elements):
        sx, sy = element_center(elements[element_idx])
    else:
        sx, sy = 80.0, 105.0
    candidates = [i for i, e in enumerate(elements) if predicate(e)]
    if not candidates:
        return element_idx
    return min(
        candidates,
        key=lambda i: (element_center(elements[i])[0] - sx) ** 2 +
                      (element_center(elements[i])[1] - sy) ** 2,
    )

def make_action(env, elements, at_idx, el_idx, utterance=''):
    action_types = env.unwrapped.action_space_config.action_types
    if at_idx == 0:
        el_idx = resolve_target_idx(elements, el_idx, is_clickable)
    else:
        el_idx = resolve_target_idx(elements, el_idx, is_input_like)

    if 0 <= el_idx < len(elements):
        e = elements[el_idx]
        cx, cy = element_center(e)
        ref = int(e.get("ref", 0))
    else:
        cx, cy = 80.0, 105.0
        ref = 0

    act = OrderedDict()
    act['ref'] = np.int64(ref if ref > 0 else 0)
    act['coords'] = np.array([cx, cy], dtype=np.float32)
    act['text'] = ''
    act['field'] = np.int64(0)
    act['key'] = np.int64(0)
    if at_idx == 0:
        if ref > 0 and ActionTypes.CLICK_ELEMENT in action_types:
            act['action_type'] = np.int64(action_types.index(ActionTypes.CLICK_ELEMENT))
        else:
            act['action_type'] = np.int64(action_types.index(ActionTypes.CLICK_COORDS))
    else:
        act['text'] = infer_type_text(utterance, elements, el_idx)
        if ref > 0 and ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT in action_types:
            act['action_type'] = np.int64(
                action_types.index(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT)
            )
        else:
            act['action_type'] = np.int64(action_types.index(ActionTypes.TYPE_TEXT))
    return act

def infer_type_text(utterance, elements, element_idx):
    elem = elements[element_idx] if 0 <= element_idx < len(elements) else {}
    elem_type = str(elem.get("type", "")).lower()
    elem_tag = str(elem.get("tag", "")).lower()
    is_password = elem_type == "password" or elem_tag == "input_password"

    quoted = re.findall(r'"([^"]*)"', utterance)
    user_q = re.search(r'username[^"]*"([^"]+)"', utterance, re.IGNORECASE)
    pass_q = re.search(r'password[^"]*"([^"]+)"', utterance, re.IGNORECASE)
    if is_password and pass_q:
        return pass_q.group(1).strip()
    if (not is_password) and user_q:
        return user_q.group(1).strip()
    if quoted:
        if is_password and len(quoted) >= 2:
            return quoted[1]
        return quoted[0]
    m = re.search(r'(?:enter|type|input|search for)\s+(.+)', utterance, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').rstrip('.')
    return utterance.strip()

def evaluate_agent(agent, task_name, device, num_episodes=50, seed=0):
    env = gym.make("miniwob/{}-v1".format(task_name), render_mode=None, wait_ms=0)
    successes = 0
    total_rewards = []
    for ep in range(num_episodes):
        try:
            obs, info = env.reset(seed=seed * 1000 + ep)
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            env = gym.make("miniwob/{}-v1".format(task_name), render_mode=None, wait_ms=0)
            obs, info = env.reset(seed=seed * 1000 + ep)
        utt = obs.get('utterance', '')
        elems = parse_dom(obs)
        feat, mask = extract_dom_features(elems, utt)
        ep_reward = 0
        for step in range(max_ep_len):
            ft = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            mt = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(ft.squeeze(0), mt.squeeze(0))
            ea = make_action(
                env, elems, action['action_type_idx'], action['element_idx'], utt
            )
            try:
                obs, reward, done, trunc, info = env.step(ea)
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass
                env = gym.make("miniwob/{}-v1".format(task_name), render_mode=None, wait_ms=0)
                ep_reward += -1.0
                break
            ep_reward += reward
            if done or trunc:
                break
            utt = obs.get('utterance', utt)
            elems = parse_dom(obs)
            feat, mask = extract_dom_features(elems, utt)
        total_rewards.append(ep_reward)
        if ep_reward > 0:
            successes += 1
    env.close()
    return successes / num_episodes, np.mean(total_rewards)

results = []

for task_name in tasks:
    for seed in seeds:
        print("\n" + "=" * 60, flush=True)
        print("PPO | Task: {} | Seed: {}".format(task_name, seed), flush=True)
        print("=" * 60, flush=True)

        env = gym.make("miniwob/{}-v1".format(task_name), render_mode=None, wait_ms=0)
        agent = PPOAgent(state_dim=256, hidden_dim=256, max_elements=64).to(device)
        pp = (list(agent.state_encoder.parameters()) +
              list(agent.action_type_head.parameters()) +
              list(agent.element_score.parameters()))
        opt_pi = torch.optim.Adam(pp, lr=lr)
        opt_v = torch.optim.Adam(list(agent.value_head.parameters()), lr=lr)

        for t in range(num_batches):
            paths, ep_rewards, episode, steps = [], [], 0, 0
            while steps < batch_size:
                try:
                    obs, info = env.reset(seed=seed + episode)
                except Exception:
                    
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = gym.make("miniwob/{}-v1".format(task_name), render_mode=None, wait_ms=0)
                    obs, info = env.reset(seed=seed + episode)
                utt = obs.get('utterance', '')
                elems = parse_dom(obs)
                feat, mask = extract_dom_features(elems, utt)
                ep_f, ep_m, ep_at, ep_ei, ep_lp, ep_r = [], [], [], [], [], []
                ep_rew = 0
                for step in range(max_ep_len):
                    ep_f.append(feat); ep_m.append(mask)
                    ft = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                    mt = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        at, ei, lp = agent.act(ft, mt, return_log_prob=True)
                    ea = make_action(env, elems, at, ei, utt)
                    try:
                        obs, reward, done, trunc, info = env.step(ea)
                    except Exception:
                        
                        try:
                            env.close()
                        except Exception:
                            pass
                        env = gym.make("miniwob/{}-v1".format(task_name), render_mode=None, wait_ms=0)
                        ep_r.append(-1.0)
                        ep_rew += -1.0
                        ep_rewards.append(ep_rew)
                        steps += 1
                        break
                    ep_at.append(at); ep_ei.append(ei); ep_lp.append(lp); ep_r.append(reward)
                    ep_rew += reward; steps += 1
                    if done or trunc:
                        ep_rewards.append(ep_rew)
                        break
                    if steps >= batch_size:
                        ep_rewards.append(ep_rew)
                        break
                    utt = obs.get('utterance', utt)
                    elems = parse_dom(obs)
                    feat, mask = extract_dom_features(elems, utt)
                else:
                    
                    ep_rewards.append(ep_rew)

                
                valid_len = min(len(ep_f), len(ep_m), len(ep_at), len(ep_ei), len(ep_lp), len(ep_r))
                if valid_len == 0:
                    episode += 1
                    continue

                paths.append({
                    'features': np.array(ep_f[:valid_len]),
                    'masks': np.array(ep_m[:valid_len]),
                    'action_types': np.array(ep_at[:valid_len]),
                    'element_idxs': np.array(ep_ei[:valid_len]),
                    'old_logprobs': np.array(ep_lp[:valid_len]),
                    'reward': np.array(ep_r[:valid_len]),
                })
                episode += 1

            if not paths:
                continue

            af = np.concatenate([p['features'] for p in paths])
            am = np.concatenate([p['masks'] for p in paths])
            aat = np.concatenate([p['action_types'] for p in paths])
            aei = np.concatenate([p['element_idxs'] for p in paths])
            olp = np.concatenate([p['old_logprobs'] for p in paths])
            rets = get_returns(paths, gamma)
            adv = calculate_advantage(rets, af, am, agent, device)
            for k in range(update_freq):
                update_value(agent, opt_v, af, am, rets, device)
                update_policy(agent, opt_pi, af, am, aat, aei, adv, olp, eps_clip, device,
                              entropy_coeff=entropy_coeff)

            if (t + 1) % 10 == 0:
                avg = np.mean(ep_rewards)
                print("  Batch {}/{}: avg_reward={:.3f} ({} eps)".format(
                    t + 1, num_batches, avg, len(ep_rewards)), flush=True)

        env.close()

        
        success_rate, mean_reward = evaluate_agent(agent, task_name, device, num_episodes=50, seed=seed)
        print("  EVAL: success_rate={:.0%} mean_reward={:.3f}".format(success_rate, mean_reward), flush=True)

        
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "ppo_{}_s{}.pt".format(task_name, seed))
        torch.save(agent.state_dict(), model_path)

        results.append({
            'method': 'ppo',
            'task': task_name,
            'seed': seed,
            'success_rate': round(success_rate, 4),
            'mean_reward': round(float(mean_reward), 6),
            'num_episodes': 50,
        })

print("\n\n" + "=" * 70, flush=True)
print("FINAL PPO RESULTS", flush=True)
print("=" * 70, flush=True)
print("{:<22} {:>8} {:>8} {:>8} {:>8}".format("Task", "s42", "s123", "s456", "Avg"), flush=True)
print("-" * 56, flush=True)
for task in tasks:
    task_results = [r for r in results if r['task'] == task]
    rates = [r['success_rate'] * 100 for r in task_results]
    seed_strs = ["{:.0f}%".format(r) for r in rates]
    while len(seed_strs) < 3:
        seed_strs.append("--")
    avg = np.mean(rates) if rates else 0
    print("{:<22} {:>8} {:>8} {:>8} {:>8.0f}%".format(task, *seed_strs, avg), flush=True)

results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'ppo_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to", results_path, flush=True)
