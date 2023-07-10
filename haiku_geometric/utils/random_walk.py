import numpy as np
from numba import njit
import jax
import jax.numpy as jnp

@njit
def random_walk(
        senders,
        receivers,
        walk_length,
        p: float = 1,
        q: float = 1,
        num_nodes=None):
    """
    Random walk on the input graph.

    Args:
        senders (jnp.ndarray): Array of sender nodes.
        receivers (jnp.ndarray): Array of receiver nodes.
        walk_length (int): Length of the random walk.
        p (float): Likelihood of returning to a previous node in the walk  
            (default: 1).
        q (float): Parameter to interpolate between breadth-first strategy and depth-first strategy 
            (default: 1).
        num_nodes (int): Number of nodes in the graph 
            (default: None).
    """
    if num_nodes is None:
        num_nodes = np.max(np.maximum(senders, receivers)) + 1
    # for each node we generate a random walk
    walks = []
    if p == 1. and q == 1.:
        for t in range(num_nodes):
            walks.append(_uniform_random_walk(senders, receivers, walk_length, t))
    else:
        for t in range(num_nodes):
            walks.append(_random_walk(senders, receivers, walk_length, p, q, t))
    return walks

@njit
def _random_neighbor(t, senders, receivers):
    """sample a random neighbor of t
    """
    # Get the neighbors of t
    neighbors = receivers[senders == t]

    # Sample a random neighbor
    return np.random.choice(neighbors, replace=False)

@njit
def _is_neighbour(t, t_prime, senders, receivers):
    """check if t_prime is a neighbour of t
    """
    return np.any(np.logical_and(senders == t, receivers == t_prime))


@njit
def _uniform_random_walk(senders, receivers, walk_length, t):
    """sample a random walk starting from t
    """
    # Initialize the walk
    walk = np.empty((walk_length + 1,), dtype=np.int32)
    walk[0] = t
    for j in range(1, walk_length + 1):
        new_node = _random_neighbor(walk[j - 1], senders, receivers)
        walk[j] = new_node
    return walk


# Adapted from: https://louisabraham.github.io/articles/node2vec-sampling.html
@njit
def _random_walk(senders, receivers, walk_length, p, q, t):
    """sample a random walk starting from t
    """
    # Normalize the weights to compute rejection probabilities
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    # Initialize the walk
    walk = np.empty((walk_length + 1,), dtype=np.int32)
    walk[0] = t
    walk[1] = _random_neighbor(t, senders, receivers)
    for j in range(2, walk_length + 1):  # we don't count the first node as part of the walk
        while True:
            new_node = _random_neighbor(walk[j - 1], senders, receivers)
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:  # back to previous node
                    break
            elif _is_neighbour(walk[j - 2], new_node, senders, receivers):
                if r < prob_1:  # distance 1
                    break
            elif r < prob_2:  # distance 2
                break
        walk[j] = new_node
    return walk

def _hk_random_walk(rowptr, col, start, walk_length, p, q):
    '''
    This is a Haiku transform-compatible version of the random walk function
    However, it needs further testing before integration
    TODO: test and integrate this function / compare performance with numpy+numba
    '''
    walks = jnp.zeros((len(start), walk_length), dtype=jnp.int32)
    rowptr = rowptr.astype(jnp.int32)
    col = col.astype(jnp.int32)
    for i, start_node in enumerate(start):
        cur_node = start_node
        for j in range(walk_length):
            neighbors = col[rowptr[jnp.int64(cur_node):jnp.int64(cur_node) + 1]]
            probs = jnp.ones_like(neighbors, dtype=jnp.float32)
            probs *= 1.0 / p
            mask = (neighbors != cur_node)
            probs = probs.at[mask].set(probs[mask] * q)
            probs = jax.nn.softmax(probs)
            next_node = jax.random.choice(jax.random.PRNGKey(0), neighbors, p=probs)
            walks = walks.at[(i, j)].set(cur_node)
            cur_node = next_node
    return walks

