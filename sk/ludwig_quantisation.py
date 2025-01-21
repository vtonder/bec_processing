import numpy as np
import matplotlib.pyplot as plt


def spectral_kurtosis(S1, S2, M, Nd=1):
    return (M * Nd + 1) / (M - 1) * ((M * S2) / (S1 ** 2) - 1)


M = 1024
K = 1000 #75000
xr = np.random.normal(size=M * K)
xi = np.random.normal(size=M * K)
x = xr + 1j * xi
p = np.abs(x.reshape(M, -1)) ** 2
S1 = p.sum(axis=0)
S2 = (p * p).sum(axis=0)
sk = spectral_kurtosis(S1, S2, M)

n_levels = 7 #255
max_level = 3.0
mid = n_levels // 2 + 1
levels = max_level * (1 + np.arange(n_levels) - mid) / mid
edges = 0.5 * (levels[1:] + levels[:-1])

print("mid   :", mid)
print("levels:", levels)
print("edges :", edges)

xrq = levels[edges.searchsorted(xr)]
xiq = levels[edges.searchsorted(xi)]
xq = xrq + 1j * xiq
pq = np.abs(xq.reshape(M, -1)) ** 2
S1q = pq.sum(axis=0)
S2q = (pq * pq).sum(axis=0)
skq = spectral_kurtosis(S1q, S2q, M)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].hist(xr, bins=512, density=True, label='continuous')
ax[0].hist(xrq, bins=512, density=True, label=f'{n_levels}-level quantisation')
ax[0].legend()
ax[0].set_title('Voltage (real component)')
ax[1].hist(sk, bins=256, density=True, label='continuous')
ax[1].hist(skq, bins=256, density=True, label=f'{n_levels}-level quantisation')
ax[1].legend()
ax[1].set_title(f'SK (M = {M})')
plt.show()
#fig.savefig('sk_v_quant_sim.png', dpi=300)

N = 25
M = 1500

pn = p.reshape(M, -1, N).sum(axis=-1)
S1n = pn.sum(axis=0)
S2n = (pn * pn).sum(axis=0)
skn = spectral_kurtosis(S1n, S2n, M, N * 2 * 0.5)

# From Helbourg's SK of quantised signals paper
p_levels = 2 * N * np.array([1.000, 1.259, 1.585, 1.995]) / 1.41
p_edges = 0.5 * (p_levels[1:] + p_levels[:-1])
pnq = p_levels[p_edges.searchsorted(pn)]
S1nq = pnq.sum(axis=0)
S2nq = (pnq * pnq).sum(axis=0)
sknq = spectral_kurtosis(S1nq, S2nq, M, N * 2 * 0.5)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].hist(pn.ravel(), bins=100, density=True, label='continuous')
ax[0].hist(pnq.ravel(), bins=100, density=True, label=f'4-level quantisation')
ax[0].legend()
ax[0].set_title(f'Raw power (N = {N})')
ax[1].hist(skn, bins=100, density=True, label='continuous')
ax[1].hist(sknq, bins=100, density=True, label=f'4-level quantisation')
ax[1].legend()
ax[1].set_title(f'SK (M = {M})')
fig.savefig('sk_p_quant_sim.png', dpi=300)