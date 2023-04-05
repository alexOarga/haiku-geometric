import jax.numpy as jnp

from haiku_geometric.utils import eigv_magnetic_laplacian

def test_mag_laplacian():
    # create synthetic graph
    senders = []
    receivers = []
    weights = []
    nodes = set()

    NODES = 10

    for i in range(NODES - 1):
        senders.append(i)
        receivers.append(i + 1)
        nodes.add(i)
        nodes.add(i + 1)
        weights.append(1)

    eigenvalues, eigenvectors = eigv_magnetic_laplacian(
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=len(nodes),
        k=5,
        k_excl=0,
        q=0.25,
        q_absolute=False,
        norm_comps_sep=False,
        l2_norm=True,
        sign_rotate=True,
        use_symmetric_norm=False,
    )

    expected_eigvalues = jnp.array([-3.0105465e-09,  9.7886987e-02,  3.8196611e-01,  8.2442987e-01, 1.3819667e+00])
    expected_eigvectors = jnp.array(
        [[3.16227555e-01 - 3.6379788e-12j, 4.41707790e-01 - 3.5527137e-15j, 4.25325692e-01 - 4.5474735e-13j,
          3.98470372e-01 + 0.0000000e+00j, 3.61803651e-01 + 3.0521733e-08j],
         [3.11423481e-01 - 5.4912381e-02j, 3.92416805e-01 - 6.9193669e-02j, 2.58872360e-01 - 4.5646161e-02j,
          6.88969791e-02 - 1.2148394e-02j, -1.36097044e-01 + 2.3997584e-02j],
         [2.97156900e-01 - 1.0815634e-01j, 2.97157079e-01 - 1.0815630e-01j, 2.59114671e-07 - 9.5603774e-08j,
          -2.97156811e-01 + 1.0815621e-01j, -4.20243472e-01 + 1.5295614e-01j],
         [2.73861438e-01 - 1.5811406e-01j, 1.75829887e-01 - 1.0151538e-01j, -2.27647990e-01 + 1.3143262e-01j,
          -3.82530123e-01 + 2.2085385e-01j, -1.19681917e-01 + 6.9098368e-02j],
         [2.42244676e-01 - 2.0326754e-01j, 5.35923541e-02 - 4.4969294e-02j, -3.25818002e-01 + 2.7339378e-01j,
          -1.55530587e-01 + 1.3050570e-01j, 2.77157575e-01 - 2.3256284e-01j],
         [2.03267336e-01 - 2.4224481e-01j, -4.49690670e-02 + 5.3592138e-02j, -2.73393750e-01 + 3.2581809e-01j,
          1.30505651e-01 - 1.5553054e-01j, 2.32562751e-01 - 2.7715749e-01j],
         [1.58113897e-01 - 2.7386150e-01j, -1.01515196e-01 + 1.7582966e-01j, -1.31432727e-01 + 2.2764817e-01j,
          2.20853910e-01 - 3.8253018e-01j, -6.90983310e-02 + 1.1968184e-01j],
         [1.08156167e-01 - 2.9715714e-01j, -1.08156130e-01 + 2.9715684e-01j, 3.54958019e-08 - 1.2446888e-07j,
          1.08156309e-01 - 2.9715705e-01j, -1.52956083e-01 + 4.2024329e-01j],
         [5.49121983e-02 - 3.1142360e-01j, -6.91934898e-02 + 3.9241639e-01j, 4.56461273e-02 - 2.5887218e-01j,
          -1.21483551e-02 + 6.8896815e-02j, -2.39975639e-02 + 1.3609703e-01j],
         [-1.79246854e-07 - 3.1622764e-01j, 1.49316691e-07 + 4.4170755e-01j, -1.38929863e-08 - 4.2532548e-01j,
          1.46173988e-08 + 3.9847037e-01j, -7.97035682e-09 - 3.6180329e-01j]])

    assert jnp.allclose(eigenvalues, expected_eigvalues, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(eigenvectors, expected_eigvectors, rtol=1e-04, atol=1e-04)

