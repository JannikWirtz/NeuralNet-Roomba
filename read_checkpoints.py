import numpy as np
import matplotlib.pyplot as plt

colors = {1: 'r', 2: 'g', 3: 'b'}

fig = plt.figure(figsize=(10, 7))
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Fitness over generations")
ax3 = plt.gca()

avg_avg = [0]*50
avg_best = [0]*50

for i in range(1, 4):
    folder_name = f'checkpoints/GA_bkp_mut=0.2_run{i}'
    avg = np.load(f'{folder_name}/average_fit.npy')
    best = np.load(f'{folder_name}/best_fit.npy')
    avg_avg = np.add(avg_avg, avg)
    avg_best = np.add(avg_best, best)
    if i == 1:
        avg_stack = avg
        best_stack = best
    else: 
        avg_stack = np.vstack((avg_stack, avg))
        best_stack = np.vstack((best_stack, best))
    x = np.arange(len(avg))

    ax3.set_ylim([np.min(best), np.max(best)*1.1])
    plt.plot(avg, f'{colors[i]}', label=f'Run {i} average fitness')
    plt.plot(best, f'{colors[i]}--', label=f'Run {i} max fitness')
    plt.legend()

plt.savefig('fitness_over_time_comparison.png')
plt.clf()
fig = plt.figure(figsize=(10, 7))
plt.xlabel("Generations")
plt.ylabel("diversity")
plt.title("diversity over generations")


avg_avg = np.divide(avg_avg, 3)
avg_best = np.divide(avg_best, 3)
std_avg = np.std(avg_stack,axis=0)
std_best = np.std(best_stack,axis=0)

fig = plt.figure(figsize=(10, 7))
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Fitness over generations")
ax3 = plt.gca()
ax3.set_ylim([np.min(best), np.max(best)*1.1])
plt.plot(avg_avg, 'k', label=f'Average fitness over 3 runs')
ax3.errorbar(x, avg_avg,
                yerr=std_avg,
                fmt='none',
                ecolor='k',
                capsize=2)
plt.plot(avg_best, 'k--', label=f'Max fitness over 3 runs')
ax3.errorbar(x, avg_best,
                yerr=std_best,
                fmt='none',
                ecolor='k',
                capsize=2)
plt.legend()
plt.savefig('fitness_over_time_merged.png')
plt.clf()
fig = plt.figure(figsize=(10, 7))
plt.xlabel("Generations")
plt.ylabel("diversity")
plt.title("diversity over generations")


for i in range(1, 4):
    folder_name = f'checkpoints/GA_bkp_mut=0.2_run{i}'
    diversity = np.load(f'{folder_name}/diversity.npy')
    ax3 = plt.gca()
    ax3.set_ylim([0, np.max(diversity)*1.1])
    plt.plot(diversity, f'{colors[i]}', label=f'Run {i} diversity')
    plt.legend()

plt.savefig('diversity_comparison.png')
