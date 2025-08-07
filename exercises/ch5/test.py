import numpy as np
for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
    # qstep = 12 * q_scale + 3
    qstep = 3 * np.log(5.3 * q_scale + 1) + 3
    print(f'qstep={qstep} when q_scale={q_scale}')
    # lambda_rd = 0.8 * (q_scale ** 12.8)
    # if q_scale <
    # lambda_rd = 0.057 * (q_scale ** 1.2)
    # print(f'lambda={lambda_rd} when q_scale={q_scale}')

