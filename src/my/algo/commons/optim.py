import math

def cosine_learning_rates(init_lr: float, T_max: int, lr_min: float = 0.0) -> list[float]:
    learning_rates = []
    for epoch in range(T_max):
        lr = lr_min + 0.5 * (init_lr - lr_min) * (1 + math.cos(epoch / T_max * math.pi))
        learning_rates.append(lr)
    return learning_rates
