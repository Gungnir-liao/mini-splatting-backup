# =========================================================================
#  基准调度器 (Baseline Schedulers)
#  对应论文中的 Static + FIFO 和 Static + EDF 策略
#  特点：不支持降质 (Fixed q=1.0)，仅做排序，没有准入控制
# =========================================================================

class BaselineScheduler:
    def __init__(self, mode='EDF'):
        """
        初始化基准调度器
        :param mode: 调度策略，'FIFO' (先来先服务) 或 'EDF' (最早截止期优先)
        """
        if mode not in ['FIFO', 'EDF']:
            raise ValueError("Mode must be either 'FIFO' or 'EDF'")
            
        self.mode = mode 
        self.name = f"{mode}_Static"
        
    def schedule(self, queue, current_time):
        """
        执行调度决策
        :param queue: 当前就绪队列 (List of Frame objects)
        :param current_time: 当前系统时间
        :return: (best_frame, quality) 选中的帧以及对应的质量参数
        """
        if not queue:
            return None, 1.0
            
        # 1. 排序策略
        candidates = list(queue)
        
        if self.mode == 'FIFO':
            # 按到达时间 (Release time, R) 升序排序
            candidates.sort(key=lambda f: f.r)
        elif self.mode == 'EDF':
            # 按截止时间 (Deadline, D) 升序排序
            candidates.sort(key=lambda f: f.d)
            
        # 2. 选择队头任务
        # 基准算法没有准入控制 (Admission Control)，即使预知会超时也会硬着头皮执行
        best_frame = candidates[0]
        
        # 3. 静态质量 (无弹性)
        # Static 方案始终尝试以最高质量 (q=1.0) 渲染
        # 如果因为耗时过长错过了截止期，Simulator 会在执行后将其标记为 Miss / DROPPED
        return best_frame, 1.0