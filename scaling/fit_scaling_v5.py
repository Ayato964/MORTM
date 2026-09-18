"""v5格子25点を Chinchilla Approach-3 (log空間Huber損失) で厳密フィット。
L(N,D) = E + A*N^-alpha + B*D^-beta
logL_pred = logsumexp([logA-alpha*logN, logB-beta*logD, logE])  を log L_obs に Huber(δ)で当てる。
scipy無しなので Hooke-Jeeves パターンサーチ + 多点初期化。ブートストラップでCI。
"""
import numpy as np

# 実測パラメータ数(本体build実測, 非依存にtotalを使用)
Ns = {'10M': 11.16e6, '20M': 22.74e6, '40M': 41.62e6, '80M': 78.95e6, '160M': 165.01e6}
# D は wandb 実測トークン数 (axis/tokens)。80M行は LR=1.2e-3 撮り直し版。
# ★最終確定セット: D=200M列を除外(LRノイズ大), 300Mは不使用。5サイズ×4データ=20点。
# (name, D_tokens(実測), val_loss)  ※ Dラベルは tokens から自明
PTS_FULL = [
 ('80M',3169151592,1.3016268),('80M',1598123771,1.3788648),('80M',790867692,1.4921025),('80M',396579590,1.7159255),
 ('10M',3167544647,1.4474243),('160M',3169474327,1.2789347),('160M',1583595307,1.3584166),('160M',791108357,1.4789590),('160M',396700430,1.6731815),
 ('40M',3169003081,1.3357986),('40M',1583176409,1.4060833),('40M',790880502,1.5242585),('40M',396564452,1.8809127),
 ('20M',3167532701,1.3708837),('20M',1582345385,1.4450526),('20M',790632360,1.5509951),('20M',396444292,1.7914799),
 ('10M',1582239527,1.5112664),('10M',790566902,1.6373227),('10M',396492801,1.8841320),
]
lN = np.array([np.log(Ns[s]) for s, t, l in PTS_FULL])
lD = np.array([np.log(t) for s, t, l in PTS_FULL])
lL = np.array([np.log(l) for s, t, l in PTS_FULL])
PTS = PTS_FULL  # 後方互換
DELTA = 1e-3


def logsumexp(rows):
    m = np.max(rows, axis=0)
    return m + np.log(np.sum(np.exp(rows - m), axis=0))


def huber(r, d=DELTA):
    a = np.abs(r)
    return np.where(a <= d, 0.5*r*r, d*(a - 0.5*d))


def loss(theta, lN=lN, lD=lD, lL=lL):
    a, b, e, al, be = theta            # a=logA, b=logB, e=logE
    pred = logsumexp(np.stack([a - al*lN, b - be*lD, e*np.ones_like(lN)]))
    return np.mean(huber(pred - lL))


def hooke_jeeves(f, x0, step=0.3, tol=1e-7, shrink=0.5, itmax=300):
    x = np.array(x0, float); fx = f(x); n = len(x); s = step
    for _ in range(itmax):
        improved = False
        for i in range(n):
            for d in (+s, -s):
                y = x.copy(); y[i] += d; fy = f(y)
                if fy < fx: x, fx = y, fy; improved = True
        if not improved:
            s *= shrink
            if s < tol: break
    return x, fx


def fit(lN=lN, lD=lD, lL=lL, starts=None):
    best = None
    Lobs = np.exp(lL)
    if starts is None:
        starts = [(al0, be0, E0) for al0 in np.arange(0.2, 1.3, 0.25)
                  for be0 in np.arange(0.2, 1.3, 0.25) for E0 in (0.9, 1.2)]
    for al0, be0, E0 in starts:
        X = np.stack([np.exp(-al0*lN), np.exp(-be0*lD)], 1)
        c, *_ = np.linalg.lstsq(X, Lobs - E0, rcond=None)
        if (c <= 0).any():
            c = np.array([1.0, 1.0])
        x0 = [np.log(max(c[0],1e-6)), np.log(max(c[1],1e-6)), np.log(E0), al0, be0]
        x, fx = hooke_jeeves(lambda t: loss(t, lN, lD, lL), x0)
        if best is None or fx < best[1]:
            best = (x, fx)
    return best[0]


def fit_warm(theta0, lN, lD, lL):
    """ブートストラップ用: 点推定からウォームスタートで1回だけHJ。"""
    x, _ = hooke_jeeves(lambda t: loss(t, lN, lD, lL), theta0, step=0.1, tol=1e-6, itmax=200)
    return x


def report(theta, tag=''):
    a, b, e, al, be = theta
    A, B, E = np.exp(a), np.exp(b), np.exp(e)
    pred = np.exp(logsumexp(np.stack([a-al*lN, b-be*lD, e*np.ones_like(lN)])))
    rmse = np.sqrt(np.mean((pred-np.exp(lL))**2))
    print(f'{tag}L = {E:.4f} + {A:.4g}/N^{al:.4f} + {B:.4g}/D^{be:.4f}   RMSE={rmse:.4f} nats')
    print(f'{tag}allocation: N_opt~C^{be/(al+be):.3f}, D_opt~C^{al/(al+be):.3f}')
    return A, B, E, al, be, rmse


if __name__ == '__main__':
    th = fit()
    print('=== 25点 厳密フィット (Chinchilla Approach-3, Huber log) ===')
    A,B,E,al,be,rmse = report(th)

    # 残差表
    a,b,e,_,_ = th
    pred = np.exp(logsumexp(np.stack([a-al*lN, b-be*lD, e*np.ones_like(lN)])))
    print('\n--- 残差 (obs-pred) ---')
    for (na,nb,l),p in zip(PTS, pred):
        print(f'  {na:>4} x {nb:>4}: obs={l:.4f} pred={p:.4f} d={l-p:+.4f}')

    # bootstrap CI
    rng = np.random.RandomState(0)
    AL=[];BE=[];EE=[];PA=[]
    for _ in range(200):
        idx = rng.randint(0, len(PTS), len(PTS))
        try:
            t = fit(lN[idx], lD[idx], lL[idx])
            AL.append(t[3]); BE.append(t[4]); EE.append(np.exp(t[2])); PA.append(t[4]/(t[3]+t[4]))
        except Exception:
            pass
    AL=np.array(AL);BE=np.array(BE);EE=np.array(EE);PA=np.array(PA)
    def ci(x): return f'{np.median(x):.3f} [{np.percentile(x,16):.3f}, {np.percentile(x,84):.3f}]'
    print(f'\n=== bootstrap n={len(AL)} (中央値[16-84%]) ===')
    print(f'alpha       = {ci(AL)}')
    print(f'beta        = {ci(BE)}')
    print(f'E           = {ci(EE)}')
    print(f'N_opt exp a = {ci(PA)}   (Chinchilla 0.46)')

    # compute-optimal D/N at several C
    print('\n=== compute-optimal (各C下で 6ND=C 制約のL最小) ===')
    for C in [1e16,1e17,1e18,1e19,1e20]:
        Ng=np.geomspace(1e6,1e11,6000); Dg=C/(6*Ng)
        Lg=E+A*Ng**-al+B*Dg**-be; i=np.argmin(Lg)
        print(f'  C={C:.0e}: N*={Ng[i]:.2e} D*={Dg[i]:.2e} D/N={Dg[i]/Ng[i]:.0f} L*={Lg[i]:.4f}')
