import random
import jax

def random_key():
    return jax.random.PRNGKey(random.randint(1, 9999999999))
