import pandas as pd
import numpy as np
import math
import heapq
import time

# ================= 配置区域 =================
W_R = 1.0  
W_Q = 1.0  
Q_SCALE = 1.0  # [修改] 设为 1.0，直接使用原始质量
# ===========================================

class Frame:
    def __init__(self, row):
        self.fid = int(row['Frame_ID'])
        self.uid = int(row['User_ID'])
        self.r = row['R']
        self.d = row['D']
        self.pred_cost = row['Pred_Cost']
        self.real_cost = row['Real_Cost']
        self.g_params = (row['Param_a'], row['Param_b'], row['Param_c'])
        self.status = 'WAITING' 

    def calculate_render_time(self, q):
        a, b, c = self.g_params
        scale = a * (q**2) + b * q + c
        scale = max(0.01, scale)
        return self.real_cost * scale

    def __repr__(self):
        return f"F(u{self.uid}_{self.fid})"

class Simulator:
    def __init__(self, trace_path):
        print(f"Loading trace: {trace_path}...")
        self.df = pd.read_csv(trace_path)
        self.all_frames = [Frame(row) for _, row in self.df.iterrows()]
        
        # 计算每个用户的会话时长 T_u
        self.user_durations = {}
        if not self.df.empty:
            grouped = self.df.groupby('User_ID')
            for uid, group in grouped:
                duration = group['D'].max() - group['R'].min()
                self.user_durations[uid] = max(1.0, duration)
        
        self.reset()

    def reset(self):
        self.current_time = 0.0
        self.ready_queue = [] 
        self.completed_history = [] 
        self.frame_cursor = 0 

    def run(self, scheduler_callback, name="Unknown", verbose=False):
        print(f"\n--- Simulation Start: {name} ---")
        self.reset()
        
        total_processed = 0
        total_dropped = 0
        start_time = time.time()

        while self.frame_cursor < len(self.all_frames) or self.ready_queue:
            if not self.ready_queue and self.frame_cursor < len(self.all_frames):
                next_arrival = self.all_frames[self.frame_cursor].r
                self.current_time = max(self.current_time, next_arrival)

            while self.frame_cursor < len(self.all_frames):
                f = self.all_frames[self.frame_cursor]
                if f.r <= self.current_time + 1e-9:
                    self.ready_queue.append(f)
                    self.frame_cursor += 1
                else:
                    break
            
            valid_queue = []
            for f in self.ready_queue:
                if f.d > self.current_time:
                    valid_queue.append(f)
                else:
                    f.status = 'DROPPED'
                    reason = f"QUEUE_TIMEOUT (Delay: {self.current_time - f.d:.4f}s)"
                    self.completed_history.append((f, 0.0, False, reason))
                    total_dropped += 1
                    #if verbose: print(f"❌ [Drop-Queue] {f} expired...")
            self.ready_queue = valid_queue

            if not self.ready_queue: continue

            selected_frame, quality = scheduler_callback(self.ready_queue, self.current_time)

            if selected_frame:
                self.ready_queue.remove(selected_frame)
                actual_duration = selected_frame.calculate_render_time(quality)
                finish_time = self.current_time + actual_duration
                
                if finish_time <= selected_frame.d:
                    selected_frame.status = 'SUCCESS'
                    self.completed_history.append((selected_frame, quality, True, "SUCCESS"))
                    total_processed += 1
                else:
                    selected_frame.status = 'DROPPED'
                    reason = f"EXEC_FAIL (Over: {finish_time - selected_frame.d:.4f}s)"
                    self.completed_history.append((selected_frame, quality, False, reason))
                    total_dropped += 1
                    if verbose: print(f"⚠️ [Drop-Exec] {selected_frame} late...")
                self.current_time = finish_time
            else:
                self.current_time += 0.001

        elapsed_time = time.time() - start_time
        
        utility, details = self.calculate_utility()
        
        print(f"Simulation Done ({elapsed_time:.2f}s).")
        print(f"  Processed: {total_processed}")
        print(f"  Dropped: {total_dropped}")
        print(f"  Total Utility: {utility:.4f}")
        return utility

    def calculate_utility(self):
        """
        [对称 Log-Throughput 效用计算]
        U_rate = w_r * log(1 + Count / Duration)
        U_qual = w_q * log(1 + Sum(q * Scale) / Duration)
        Scale = 1.0 (原始质量)
        """
        user_stats = {} 
        
        for frame, q, is_success, _ in self.completed_history:
            if not is_success: continue
            uid = frame.uid
            if uid not in user_stats: user_stats[uid] = {'count': 0, 'sum_q_scaled': 0.0}
            
            user_stats[uid]['count'] += 1
            # 累加原始质量值
            user_stats[uid]['sum_q_scaled'] += (max(0.01, q) * Q_SCALE)
            
        total_utility = 0
        for uid, stats in user_stats.items():
            t_u = self.user_durations.get(uid, 1.0)
            
            # 1. 帧率吞吐 (FPS)
            throughput_r = stats['count'] / t_u
            u_rate = W_R * math.log(1 + throughput_r)
            
            # 2. 质量吞吐 (Quality Flux)
            throughput_q = stats['sum_q_scaled'] / t_u
            u_qual = W_Q * math.log(1 + throughput_q)
            
            total_utility += (u_rate + u_qual)
            
        return total_utility, user_stats