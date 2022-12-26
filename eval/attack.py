from __future__ import absolute_import, division, print_function

import importlib
import random

import numpy as np
import torch
from robustness.attacks import ATTACKS

class Attack:
    SUPPORTED_ATTACKS = ["pgd", "fgsm", "s&p", "gaussian"]

    def __init__(self, epsilon, model):
        self.epsilon = epsilon
        self.model = model
        self.gaussian = GaussianNoise(self.epsilon)
        self.snp = SnPNoise(self.epsilon)
        self.fgsm = self._create_adversarial_attack("fgsm")(epsilon=self.epsilon)
        self.pgd = self._create_adversarial_attack("pgd")(epsilon=self.epsilon,
                                                          alpha=2 / 255,
                                                          iterations=40,
                                                          random_start=True)

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def __call__(self, x, y):
        dict_ = {}
        with torch.enable_grad():
            dict_["pgd": {"x_adv": self.pgd(x, y, self.model).detach()}]
            dict_["fgsm": {"x_adv": self.fgsm(x, y, self.model).detach()}]
        dict_["s&p": {"x_adv": self.snp(x).detach()}]
        dict_["gaussian": {"x_adv": self.gaussian(x).detach()}]

        for a in self.SUPPORTED_ATTACKS:
            dict_[a: {"epsilon": self._compute_attack_strength(x, dict_[a]["x_adv"])}]
        return dict_

    def _create_adversarial_attack(self, name):
        module = importlib.import_module(ATTACKS[name].classpath)
        attack = getattr(module, ATTACKS[name].classname)
        return attack

    def apply_attack(self, attack, x, y):
        assert isinstance(attack, str), "attack should be of type string"
        assert attack in self.SUPPORTED_ATTACKS, f"The attack type {attack} is not supported. " \
            f"Please choose from {self.SUPPORTED_ATTACKS}."

        with torch.enable_grad():
            if attack == "pgd":
                x_adv = self.pgd(x, self.model, y)
            if attack == "fgsm":
                x_adv = self.fgsm(x, self.model, y)
        if attack == "s&p":
            x_adv = self.snp(x)
        if attack == "gaussian":
            x_adv = self.gaussian(x)
        return x_adv.detach(), self._compute_attack_strength(x, x_adv).detach()

    def _compute_attack_strength(self, x, x_adv):
        r = x_adv - x
        return torch.sqrt((r * r).mean())


class GaussianNoise:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, x):
        noise = torch.normal(0, self.epsilon, size=x.shape).cuda()
        x = torch.clamp(x + noise, 0, 1)
        return x


class SnPNoise:
    # See also here for explanation: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1510683
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.svp = 0.5
        self.amount = epsilon * epsilon * 4  # This is how Marvin calculated the amount of S & P
        self.s = 1
        self.p = 0

    def __call__(self, x):
        num_s = int(np.ceil(np.prod(x.shape) * self.amount * self.svp))
        num_p = int(np.ceil(np.prod(x.shape) * self.amount * (1 - self.svp)))
        x_ = x.clone()
        x_ = x_.reshape(-1)
        idx = np.random.randint(0, x_.shape[0], num_s + num_p)
        idx_s = idx[:num_s]
        idx_p = idx[num_s:]
        x_[idx_s] = self.s
        x_[idx_p] = self.p
        x_ = x_.reshape(x.shape)
        x_ = x_.clamp(0, 1)
        return x_


def main():
    pass


if __name__ == "__main__":
    main()
