import torch
import torchvision.models as models
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

model = models.resnet18().cuda()
inputs = torch.randn(1, 3, 224, 224).cuda()
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 300次中第一次测试时间过长，需要删除 by xgh 230509 20：43
# model = models.resnet18().cuda()
# device = torch.device('cuda')
# model.to(device)
# dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings = np.zeros((repetitions, 1))
# #GPU-WARM-UP：开始跑dummy example
# for _ in range(10):
#    _ = model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#   for rep in range(repetitions):
#      starter.record()
#      _ = model(dummy_input)
#      ender.record()
#      # WAIT FOR GPU SYNC
#      torch.cuda.synchronize()
#      curr_time = starter.elapsed_time(ender)
#      timings[rep] = curr_time
# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print(mean_syn)



#
# import torch
# import torch.utils.benchmark as benchmark
# model = models.resnet18().cuda()
# inputs = torch.randn(1, 3, 224, 224).cuda()
#
# timer = benchmark.Timer(
#     stmt='model(inputs)',  # 测试语句，即模型推理
#     setup='from __main__ import model, inputs',  # 环境变量
#     num_threads=1,  # 线程数
#     # num_runs=10,  # 测试次数
# )
#
# result = timer.blocked_autorange(min_run_time=1)  # 执行测试
# print(result)  # 输出结果

