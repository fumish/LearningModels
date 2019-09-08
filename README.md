# gp_related
gaussian process related works are created in this repo
# LVA
Learning library for local variational approximation (LVA).
LVA is a method to approximate the Bayesian posterior distribution by bounding a joint distribution.  

Let $p(x^n|w)\varphi(w)$ be the joint distribution,
then LVA makes the approximated distribution by

$\underline{p}_{\xi}(x^n,w) \leq p(x^n,w) \leq \overline{p}_{\eta}(x^n,w).$
