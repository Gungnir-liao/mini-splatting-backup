import argparse
import os
import time
import numpy as np
import math
from scipy.optimize import minimize
from simulator_core import Simulator

# ================= 配置区域 =================
# 权重配置
W_R = 1.0  
W_Q = 1.0 

# 安全余裕
SAFETY_MARGIN = 0.02
DEADLINE_BUFFER = 0.002

# 会话管理配置
SESSION_TIMEOUT_FACTOR = 3.0
SESSION_MIN_TIMEOUT = 0.1

DEBUG_LOG = False
# ===========================================

def g(q, a, b, c): return a * (q**2) + b * q + c
def g_prime(q, a, b): return 2 * a * q + b
def find_nearest(array, value): return array[(np.abs(array - value)).argmin()]

def run_continuous_sca(n, f_demands, C_bases, params_abc, R_range, Q_range, max_iter=20, epsilon=1e-4):
    """
    SCA 连续域优化
    修正后的目标: Maximize w_r * log(1+r) + w_q * log(1+q)
    """
    r_k = np.full(n, (R_range[0] + R_range[1]) / 2)
    q_k = np.full(n, (Q_range[0] + Q_range[1]) / 2)
    
    if params_abc.ndim == 1: params_abc = params_abc.reshape(1, 3)
    A_vals, B_vals, C_vals = params_abc[:, 0], params_abc[:, 1], params_abc[:, 2]
    capacity_limit = 1.0 - SAFETY_MARGIN

    for k in range(max_iter):
        r_old, q_old = r_k.copy(), q_k.copy()
        g_val = g(q_old, A_vals, B_vals, C_vals)
        g_prime_val = g_prime(q_old, A_vals, B_vals)
        
        # [注意] SCA 阶段包含 buffer，与 load 计算保持一致
        frame_cost = C_bases * g_val
        
        J_vals = r_old * frame_cost
        grad_r = frame_cost
        grad_q = r_old * C_bases * g_prime_val
        
        def objective(x):
            r, q = x[:n], x[n:]
            r = np.maximum(r, 1e-5)
            q = np.maximum(q, 1e-5)
            obj_r = W_R * np.log(1 + r)
            obj_q = W_Q * np.log(1 + q)
            return -np.sum(obj_r + obj_q)
        
        def objective_jac(x):
            r, q = x[:n], x[n:]
            r = np.maximum(r, 1e-5)
            q = np.maximum(q, 1e-5)
            grad_r_obj = - (W_R / (1 + r))
            grad_q_obj = - (W_Q / (1 + q))
            return np.concatenate([grad_r_obj, grad_q_obj])
        
        def linearized_constraint(x):
            r, q = x[:n], x[n:]
            approx_load = J_vals + grad_r * (r - r_old) + grad_q * (q - q_old)
            return capacity_limit - np.sum(approx_load)
        
        def constraint_jac(x):
            return np.concatenate([-grad_r, -grad_q])
        
        bounds = [(R_range[0], min(R_range[1], f_demands[i])) for i in range(n)] + \
                 [(Q_range[0], Q_range[1]) for i in range(n)]
        x0 = np.concatenate([r_old, q_old])
        cons = {'type': 'ineq', 'fun': linearized_constraint, 'jac': constraint_jac}
        
        try:
            res = minimize(objective, x0, jac=objective_jac, bounds=bounds, constraints=cons, method='SLSQP', options={'disp': False})
            if res.success: r_k, q_k = res.x[:n], res.x[n:]
            else: break
        except Exception: break
        if np.linalg.norm(np.concatenate([r_k, q_k]) - x0) < epsilon: break
    return r_k, q_k

class SessionManager:
    def __init__(self): self.sessions = {}
    def touch(self, uid, current_time, frame_params):
        self.sessions[uid] = {'last_seen': current_time, 'data': frame_params}
    def get_active_users(self, current_time):
        active, dead = [], []
        for uid, session in self.sessions.items():
            fps = session['data']['demand_fps']
            timeout = max(SESSION_MIN_TIMEOUT, SESSION_TIMEOUT_FACTOR * (1.0 / max(1, fps)))
            if current_time - session['last_seen'] <= timeout:
                active.append({'uid': uid, **session['data']})
            else: dead.append(uid)
        for uid in dead: del self.sessions[uid]
        return active

class HierarchicalScheduler:
    def __init__(self):
        self.user_targets = {} 
        self.user_history = {}
        self.session_manager = SessionManager()
        self.last_optimize_time = -1.0
        self.optimize_interval = 1.0 
        self.min_optimize_interval = 0.1
        self.R_options = np.linspace(30, 90, 7, dtype=int)
        self.Q_options = np.linspace(0.5, 1.0, 6)

    def update_user_state(self, queue, current_time):
        user_frames_map = {}
        for f in queue:
            user_frames_map.setdefault(f.uid, []).append(f)
        for uid, frames in user_frames_map.items():
            avg_pred_cost = np.mean([f.pred_cost for f in frames])
            avg_params = np.mean([f.g_params for f in frames], axis=0)
            demand_fps = int(round(1.0 / max(1e-3, frames[0].d - frames[0].r)))
            self.session_manager.touch(uid, current_time, {'pred_cost': avg_pred_cost, 'g_params': avg_params, 'demand_fps': demand_fps})

    def run_outer_loop_optimization(self, current_time, queue):
        self.update_user_state(queue, current_time)
        active_users = self.session_manager.get_active_users(current_time)
        n = len(active_users)
        if n == 0: return
        active_users.sort(key=lambda x: x['uid'])
        online_uids = [u['uid'] for u in active_users]

        c_bases = np.array([u['pred_cost'] for u in active_users])
        f_demands = np.array([u['demand_fps'] for u in active_users], dtype=int)
        params_abc = np.array([u['g_params'] for u in active_users])

        # 1. 连续优化
        r_cont, q_cont = run_continuous_sca(
            n, f_demands, c_bases, params_abc,
            (self.R_options.min(), self.R_options.max()),
            (self.Q_options.min(), self.Q_options.max())
        )
        # 2. 离散化
        r_disc = np.array([find_nearest(self.R_options, v) for v in r_cont])
        q_disc = np.array([find_nearest(self.Q_options, v) for v in q_cont])
        r_disc = np.minimum(r_disc, f_demands)

        # 3. 贪心修正
        limit = 1.0 - SAFETY_MARGIN
        while True:
            # 当前所有用户的单帧成本 (必须包含 buffer)
            frame_costs = [c_bases[i] * g(q_disc[i], *params_abc[i]) for i in range(n)]
            load = sum([r_disc[i] * c for i, c in enumerate(frame_costs)])
            
            if load <= limit: break
            
            # 找到当前贡献负载最大的用户
            idx = np.argmax([r_disc[i] * c for i, c in enumerate(frame_costs)])
            curr_r, curr_q = r_disc[idx], q_disc[idx]
            
            lower_r_opts = self.R_options[self.R_options < curr_r - 1e-5]
            lower_q_opts = self.Q_options[self.Q_options < curr_q - 1e-5]
            
            best_move = None
            best_eff = -1.0
            
            # A. 尝试降低 R
            if len(lower_r_opts) > 0:
                nxt_r = lower_r_opts[-1]
                # 成本节省: (R_old - R_new) * Cost_per_frame
                cost_saved = (curr_r - nxt_r) * frame_costs[idx]
                
                u_loss = (W_R * math.log(1 + curr_r)) - (W_R * math.log(1 + nxt_r))
                if u_loss > 0:
                    eff = cost_saved / u_loss
                    best_eff = eff
                    best_move = ('r', nxt_r)
            
            # B. 尝试降低 Q
            if len(lower_q_opts) > 0:
                nxt_q = lower_q_opts[-1]
                # 计算新的单帧成本 (同样必须包含 buffer，否则不公平)
                new_fc = c_bases[idx] * g(nxt_q, *params_abc[idx])
                # 成本节省: R * (Cost_old - Cost_new)
                cost_saved = r_disc[idx] * (frame_costs[idx] - new_fc)
                
                u_loss = (W_Q * math.log(1 + curr_q)) - (W_Q * math.log(1 + nxt_q))
                if u_loss > 0:
                    eff = cost_saved / u_loss
                    # 如果降低 Q 比降低 R 性价比更高 (eff 更大)，则选 Q
                    if eff > best_eff:
                        best_move = ('q', nxt_q)
            
            if best_move:
                if best_move[0] == 'r': r_disc[idx] = best_move[1]
                else: q_disc[idx] = best_move[1]
            else:
                # 无法再降 (理论上不应发生，除非本来就是最低配置还超标)
                break

        # 4. 应用与日志打印
        queue_uids = set(f.uid for f in queue)
        users_display = [f"{uid}{'*' if uid in queue_uids else ''}" for uid in online_uids]
        
        # 总是先打印总览
        if DEBUG_LOG:
            print(f"\n[SCA] Time: {current_time:.2f}s | Online: {n} {users_display} | Load: {load:.3f}/{limit:.2f}")
            # 打印详细表格
            print(f"{'UID':<6} | {'Demand':<6} | {'Alloc R':<7} | {'Alloc Q':<7} | {'Est.Cost':<8}")
            print("-" * 55)

        for i, uid in enumerate(online_uids):
            self.user_targets[uid] = {'target_fps': float(r_disc[i]), 'target_q': float(q_disc[i])}
            if DEBUG_LOG:
                # 重新计算一下最终成本用于展示
                final_cost = c_bases[i] * g(q_disc[i], *params_abc[i])
                print(f"{uid:<6} | {f_demands[i]:<6d} | {r_disc[i]:<7.1f} | {q_disc[i]:<7.2f} | {final_cost:.4f}")
            
        self.last_optimize_time = current_time

    def check_admission(self, frame, current_time):
        uid = frame.uid
        target = self.user_targets.get(uid, {'target_fps': 999})
        history = self.user_history.get(uid, [])
        history = [t for t in history if t > current_time - 1.0]
        self.user_history[uid] = history
        return len(history) < target['target_fps']

    def schedule(self, queue, current_time):
        self.update_user_state(queue, current_time)
        is_time = (current_time - self.last_optimize_time >= self.optimize_interval) or (self.last_optimize_time < 0)
        active = set(u['uid'] for u in self.session_manager.get_active_users(current_time))
        target_uids = set(self.user_targets.keys())
        new_user = not active.issubset(target_uids)
        cooled = (current_time - self.last_optimize_time >= self.min_optimize_interval)
        
        if new_user and not cooled:
            pass
        elif (new_user and cooled) or (is_time and active):
            self.run_outer_loop_optimization(current_time, queue)

        candidates = []
        for f in queue:
            if not self.check_admission(f, current_time): continue
            t_q = self.user_targets.get(f.uid, {'target_q': 1.0})['target_q']
            a, b, c = f.g_params
            scale = max(0.01, a * (t_q**2) + b * t_q + c)
            if current_time + f.pred_cost * scale > f.d - DEADLINE_BUFFER: continue
            candidates.append(f)
            
        if not candidates: return None, 1.0
        candidates.sort(key=lambda f: f.d)
        best = candidates[0]
        q = self.user_targets.get(best.uid, {'target_q': 1.0})['target_q']
        if best.uid not in self.user_history: self.user_history[best.uid] = []
        self.user_history[best.uid].append(current_time)
        return best, q

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="simulation_trace.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    DEBUG_LOG = args.verbose
    if os.path.exists(args.trace):
        Simulator(args.trace).run(HierarchicalScheduler().schedule, name="Proposed", verbose=args.verbose)
    else:
        print("Trace not found.")